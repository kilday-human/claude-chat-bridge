#!/usr/bin/env python3
"""
Simple webhook server for testing Zapier integration
This simulates what Zapier would receive from our webhook calls
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="GPT-Bridge Webhook Test Server")

# Store received webhooks for inspection
received_webhooks = []

@app.post("/api/webhook/usage")
async def receive_usage_webhook(request: Request):
    """Receive usage data from GPT-Bridge"""
    try:
        payload = await request.json()
        
        # Store for inspection
        webhook_data = {
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        }
        received_webhooks.append(webhook_data)
        
        print(f"📊 Received webhook: {payload.get('session_id', 'unknown')}")
        print(f"   Query: {payload.get('query', {}).get('summary', 'N/A')}")
        print(f"   Model: {payload.get('response', {}).get('model_used', 'N/A')}")
        print(f"   Response time: {payload.get('response', {}).get('response_time_ms', 0)}ms")
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "Webhook received"}
        )
        
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.get("/webhooks")
async def list_webhooks():
    """List all received webhooks"""
    return {"webhooks": received_webhooks, "count": len(received_webhooks)}

@app.get("/")
async def root():
    """Health check"""
    return {"status": "running", "webhooks_received": len(received_webhooks)}

if __name__ == "__main__":
    print("🚀 Starting webhook test server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📊 Webhook endpoint: http://localhost:8000/api/webhook/usage")
    print("📋 View webhooks: http://localhost:8000/webhooks")
    print("\nTo test with GPT-Bridge:")
    print("1. Set ZAPIER_WEBHOOK_URL=http://localhost:8000/api/webhook/usage")
    print("2. Set ENABLE_WEBHOOK=true")
    print("3. Run: python3 cli_bridge.py 'test query' --verbose")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
