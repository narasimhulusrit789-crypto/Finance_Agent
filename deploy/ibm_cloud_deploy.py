"""
IBM Code Engine deployment configuration (Kubernetes-style manifest).
Deploy with: ibmcloud ce application create --name finance-agent ...
"""

# IBM Code Engine CLI deployment script
DEPLOY_COMMANDS = """
#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  IBM Code Engine — Finance Agent Deployment Script
# ─────────────────────────────────────────────────────────────

set -e

PROJECT_NAME="finance-agent-project"
APP_NAME="finance-agent"
IMAGE_NAME="us.icr.io/${NAMESPACE}/finance-agent:latest"
REGION="us-south"

echo "=== IBM Finance Agent Deployment ==="

# 1. Login to IBM Cloud
ibmcloud login --apikey $IBM_CLOUD_API_KEY -r $REGION

# 2. Target resource group
ibmcloud target -g default

# 3. Build and push Docker image to IBM Container Registry
ibmcloud cr login
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME

# 4. Create or switch Code Engine project
ibmcloud ce project select --name $PROJECT_NAME || \\
  ibmcloud ce project create --name $PROJECT_NAME

# 5. Create secrets for sensitive environment variables
ibmcloud ce secret create --name finance-agent-secrets \\
  --from-literal WATSONX_API_KEY=$WATSONX_API_KEY \\
  --from-literal WATSONX_PROJECT_ID=$WATSONX_PROJECT_ID \\
  --from-literal SECRET_KEY=$SECRET_KEY \\
  --from-literal IBM_COS_API_KEY=$IBM_COS_API_KEY

# 6. Deploy the application
ibmcloud ce application create \\
  --name $APP_NAME \\
  --image $IMAGE_NAME \\
  --cpu 2 \\
  --memory 4G \\
  --min-scale 1 \\
  --max-scale 5 \\
  --port 8000 \\
  --env APP_ENV=production \\
  --env WATSONX_URL=https://us-south.ml.cloud.ibm.com \\
  --env WATSONX_REGION=$REGION \\
  --env VECTOR_DB_TYPE=chromadb \\
  --env REDIS_URL=redis://redis-service:6379 \\
  --env-from-secret finance-agent-secrets

# 7. Get application URL
ibmcloud ce application get --name $APP_NAME --output url

echo "=== Deployment complete ==="
"""

# IBM Code Engine Application spec (YAML equivalent)
CODE_ENGINE_SPEC = {
    "apiVersion": "serving.knative.dev/v1",
    "kind": "Service",
    "metadata": {
        "name": "finance-agent",
        "annotations": {
            "serving.knative.dev/creator": "IBM Finance Agent",
        }
    },
    "spec": {
        "template": {
            "metadata": {
                "annotations": {
                    "autoscaling.knative.dev/minScale": "1",
                    "autoscaling.knative.dev/maxScale": "5",
                }
            },
            "spec": {
                "containerConcurrency": 10,
                "timeoutSeconds": 300,
                "containers": [{
                    "image": "us.icr.io/NAMESPACE/finance-agent:latest",
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                    "ports": [{"containerPort": 8000}],
                    "env": [
                        {"name": "APP_ENV", "value": "production"},
                        {"name": "APP_PORT", "value": "8000"},
                        {"name": "WATSONX_URL", "value": "https://us-south.ml.cloud.ibm.com"},
                        {
                            "name": "WATSONX_API_KEY",
                            "valueFrom": {"secretKeyRef": {"name": "finance-agent-secrets", "key": "WATSONX_API_KEY"}},
                        },
                        {
                            "name": "WATSONX_PROJECT_ID",
                            "valueFrom": {"secretKeyRef": {"name": "finance-agent-secrets", "key": "WATSONX_PROJECT_ID"}},
                        },
                        {
                            "name": "SECRET_KEY",
                            "valueFrom": {"secretKeyRef": {"name": "finance-agent-secrets", "key": "SECRET_KEY"}},
                        },
                    ],
                    "livenessProbe": {
                        "httpGet": {"path": "/health", "port": 8000},
                        "initialDelaySeconds": 60,
                        "periodSeconds": 30,
                    },
                    "readinessProbe": {
                        "httpGet": {"path": "/health", "port": 8000},
                        "initialDelaySeconds": 10,
                        "periodSeconds": 10,
                    },
                }]
            }
        }
    }
}
