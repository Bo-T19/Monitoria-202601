import os
import math
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

import requests
import xml.etree.ElementTree as ET
import scipy
from dotenv import load_dotenv

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult

load_dotenv()

# Configurar el LLM de OpenAI
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.llm = llm

# Configurar la seguridad de la API
_raw_keys = os.environ.get("AGENT_API_KEYS", "")
API_KEYS: set[str] = {key.strip() for key in _raw_keys.split(",") if key.strip()}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not key or not key in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida o ausente.",
            headers={"WWW-Authenticate": "API Key"}
        )
    return key

# Tools para el agente
def multiply(a: int, b: int) -> int:
    """Multiplica dos números enteros y devuelve el resultado."""
    return a * b


def add(a: int, b: int) -> int:
    """Suma dos enteros y devuelve el resultado."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Resta dos enteros y devuelve el resultado."""
    return a - b


def divide(a: float, b: float) -> float:
    """Divide dos números y devuelve el resultado."""
    if b == 0:
        return float("inf")
    return a / b


def power(a: float, b: float) -> float:
    """Eleva un número a la potencia deseada."""
    return a**b


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convierte moneda usando la API Frankfurter (BCE, sin API key)."""
    url = "https://api.frankfurter.app/latest"
    params = {"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()}
    try:
        data = requests.get(url, params=params, timeout=10).json()
        rate = data.get("rates", {}).get(to_currency.upper())
        if rate is None:
            return f"No fue posible convertir {from_currency} a {to_currency}."
        return f"{amount} {from_currency.upper()} = {rate:.2f} {to_currency.upper()}"
    except Exception as exc:
        return f"Error en conversión de moneda: {exc}"


def search_web(query: str) -> str:
    """Busca un resumen en Wikipedia (es, con fallback a en)."""
    headers = {"User-Agent": "research-agent/1.0 (educational use)"}
    for lang in ("es", "en"):
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            }
            r = requests.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params=params,
                headers=headers,
                timeout=10,
            )
            hits = r.json().get("query", {}).get("search", [])
            if not hits:
                continue
            title = hits[0]["title"]
            s = requests.get(
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                headers=headers,
                timeout=10,
            )
            if s.status_code == 200:
                extract = s.json().get("extract")
                if extract:
                    return f"[Wikipedia/{lang}] {extract}"
        except Exception as exc:
            return f"Error en búsqueda web: {exc}"
    return f"No se encontró información para '{query}'."


def get_weather(location: str = "Bogota") -> str:
    """Obtiene el clima actual usando wttr.in."""
    try:
        result = requests.get(f"https://wttr.in/{location}", params={"format": "3"}, timeout=10)
        if result.status_code == 200:
            return result.text.strip()
        return f"No se pudo obtener el clima para {location}."
    except Exception as exc:
        return f"Error al obtener clima: {exc}"


def get_news(category: str = "world") -> str:
    """Obtiene las últimas noticias vía RSS de Google News (sin API key)."""
    topic_map = {
        "world": "WORLD",
        "business": "BUSINESS",
        "technology": "TECHNOLOGY",
        "sports": "SPORTS",
        "science": "SCIENCE",
        "health": "HEALTH",
        "entertainment": "ENTERTAINMENT",
    }
    topic = topic_map.get(category.lower(), "WORLD")
    url = f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=es&gl=ES&ceid=ES:es"
    try:
        result = requests.get(url, timeout=10)
        root = ET.fromstring(result.content)
        items = root.findall(".//item")[:5]
        if not items:
            return "No se encontraron noticias."
        return "\n".join(f"- {item.findtext('title', 'Sin título')}" for item in items)
    except Exception as exc:
        return f"Error al obtener noticias: {exc}"


def current_time(timezone: str = "UTC") -> str:
    """Devuelve la hora actual en la zona horaria solicitada."""
    if ZoneInfo is None:
        return f"Hora UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (ZoneInfo no disponible)"
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return f"Zona horaria inválida. Hora UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"


def calcular_integral(expression: str, limite_inferior: float, limite_superior: float) -> float:
    """Calcula la integral definida de una función de una variable x.

    expression debe ser una expresión Python válida (ej: 'x**2 + 2*x + 1').
    """
    def f(x):
        return eval(expression, {"math": math, "x": x})  # noqa: S307 – expresión controlada por el LLM

    result, _ = scipy.integrate.quad(f, limite_inferior, limite_superior)
    return result

# Configurar el agente con las herramientas
agent = FunctionAgent(
    tools=[
        FunctionTool.from_defaults(fn= multiply),
        FunctionTool.from_defaults(fn= add),
        FunctionTool.from_defaults(fn= subtract),
        FunctionTool.from_defaults(fn= divide),
        FunctionTool.from_defaults(fn= power),
        FunctionTool.from_defaults(fn= convert_currency),
        FunctionTool.from_defaults(fn= search_web),
        FunctionTool.from_defaults(fn= get_weather),
        FunctionTool.from_defaults(fn= get_news),
        FunctionTool.from_defaults(fn= current_time),
        FunctionTool.from_defaults(fn= calcular_integral),
    ],
    llm=llm,
)

# Modelos de datos para la API
class QueryRequest(BaseModel):
    query: str

class ToolStep(BaseModel):
    tool: str
    args: dict
    output: str

class QueryResponse(BaseModel):
    response: str
    trace: list[ToolStep]

# Configurar FastAPI
app = FastAPI(
    title="Agente Funcional con Llamaindex y FastAPI",
    description="Un agente que puede realizar cálculos, búsquedas web, conversiones de moneda, y más, todo a través de una API REST.",
    version="1.0.0",
)

@app.get("/health", tags=["Sistema"])
async def health():
    """Endpoint de salud para verificar que el servicio está activo."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/query", response_model=QueryResponse, tags=["Agente"])
async def query_agent(request: QueryRequest, _: str = Security(verify_api_key)):
    """Endpoint principal para enviar consultas al agente."""
    handler = agent.run(request.query)
    trace: list[ToolStep] = []


    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            salida = str(event.tool_output)
            if len(salida) > 300:
                salida = salida[:300] + "..."
            trace.append(ToolStep(tool=event.tool_name, args=dict(event.tool_kwargs), output=salida))
    
    result = await handler
    return QueryResponse(response=str(result.response), trace=trace)



