from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request
from llama_cpp import Llama

app = Flask(__name__)

# Configuración
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH", str(BASE_DIR / "models/qwen2.5-3b-instruct-q4_k_m.gguf")
)
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "1"))

# Estado global
_LLM = None
_MODEL_ERROR = None


def load_model():
    """Carga el modelo al inicio. Si falla, guarda el error."""
    global _LLM, _MODEL_ERROR
    
    print("=" * 60)
    print("🔄 CARGANDO MODELO LLM...")
    print(f"📁 Ruta: {MODEL_PATH}")
    print("=" * 60)
    
    # Verificar que existe
    if not Path(MODEL_PATH).exists():
        error_msg = f"❌ ERROR: Modelo NO encontrado en: {MODEL_PATH}"
        print(error_msg)
        _MODEL_ERROR = error_msg
        return False
    
    print("✅ Archivo del modelo encontrado")
    
    # Intentar cargar
    try:
        print(f"⚙️  Configuración: n_ctx={N_CTX}, n_threads={N_THREADS}")
        _LLM = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
        )
        print("✅ Modelo cargado exitosamente")
        print("=" * 60)
        return True
    except Exception as e:
        error_msg = f"❌ ERROR cargando modelo: {str(e)}"
        print(error_msg)
        print("=" * 60)
        _MODEL_ERROR = error_msg
        return False


@app.get("/")
def root():
    """Información del servicio"""
    if _MODEL_ERROR:
        return jsonify({
            "service": "Agricultural LLM API",
            "status": "error",
            "error": _MODEL_ERROR,
            "model_loaded": False
        }), 500
    
    return jsonify({
        "service": "Agricultural LLM API",
        "version": "1.0.0",
        "status": "ready",
        "model_loaded": _LLM is not None,
        "model_path": MODEL_PATH,
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    })


@app.get("/health")
def health():
    """Health check - FALLA si no hay modelo"""
    if _MODEL_ERROR:
        return jsonify({
            "status": "error",
            "error": _MODEL_ERROR,
            "model_loaded": False
        }), 500
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": _LLM is not None,
        "model_path": MODEL_PATH
    })


@app.post("/chat")
def chat():
    """Endpoint principal - REQUIERE modelo o falla"""
    
    # Si hay error de modelo, retornar error inmediatamente
    if _MODEL_ERROR:
        return jsonify({
            "error": "Model not available",
            "details": _MODEL_ERROR
        }), 503
    
    # Si no hay modelo cargado (no debería pasar)
    if _LLM is None:
        return jsonify({
            "error": "Model not loaded",
            "details": "LLM instance is None"
        }), 503
    
    try:
        # Parsear request
        payload = request.get_json(force=True)
        system_prompt = payload.get("system", "Eres un asistente agrónomo.")
        context = payload.get("context", {})
        max_tokens = int(payload.get("max_tokens", 300))
        
        mensaje = context.get("mensaje", "")
        print(f"📨 Request: {mensaje[:50]}...")
        
        # Preparar mensajes
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        
        # Generar respuesta con el modelo REAL
        print("🤖 Generando respuesta...")
        response = _LLM.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        
        content = response["choices"][0]["message"]["content"] or ""
        print(f"✅ Respuesta generada: {len(content)} caracteres")
        
        return jsonify({
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Error procesando request: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            "error": error_msg,
            "type": type(e).__name__
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INICIANDO SERVIDOR - Agricultural LLM API")
    print("=" * 60)
    
    # PASO 1: Cargar modelo ANTES de levantar servidor
    model_loaded = load_model()
    
    if not model_loaded:
        print("⚠️  ADVERTENCIA: Servidor iniciará SIN modelo cargado")
        print("⚠️  Todas las peticiones /chat fallarán con error 503")
    
    # PASO 2: Levantar servidor
    port = int(os.getenv("PORT", 8080))
    print(f"🌐 Servidor escuchando en puerto {port}")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
