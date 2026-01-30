# Production Deployment Guide
## TechGear Electronics Customer Support Chatbot

This guide covers deploying the chatbot to production environments including Docker, cloud platforms (AWS, GCP, Azure), and configuration best practices.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Logging](#monitoring--logging)
6. [Security](#security)
7. [Troubleshooting](#troubleshooting)
8. [Operations](#operations)

---

## Prerequisites

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.10+ (for local development)
- Git

### Required Accounts
- Google Cloud Platform (for Gemini API)
- Cloud provider account (AWS/GCP/Azure) for cloud deployment
- Domain name (optional, for HTTPS)

### API Keys
- Google Gemini API key
- Cloud provider credentials
- Third-party service keys (monitoring, logging)

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/vijayshrimali/Customer-Support-Chatbot.git
cd Customer-Support-Chatbot
```

### 2. Configure Environment Variables

Copy the production template:

```bash
cp .env.production .env
```

Edit `.env` and configure:

```bash
# Required
GEMINI_API_KEY=your_actual_gemini_api_key

# Optional (with defaults)
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=4
RATE_LIMIT_PER_MINUTE=60
```

### 3. Generate Security Keys

```bash
# Generate API key
python -c "import secrets; print('API_KEY:', secrets.token_urlsafe(32))"

# Generate JWT secret
python -c "import secrets; print('JWT_SECRET:', secrets.token_urlsafe(64))"
```

Add these to your `.env` file.

---

## Docker Deployment

### Option 1: Docker Compose (Recommended)

Deploy all services (API, Redis, Prometheus, Grafana):

```bash
# Deploy
./deployment/deploy.sh compose

# Or manually
docker-compose up -d --build
```

### Option 2: Standalone Docker

Deploy only the API service:

```bash
# Deploy
./deployment/deploy.sh docker

# Or manually
docker build -t techgear-chatbot-api .
docker run -d \
  --name techgear-chatbot \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/logs:/app/logs \
  techgear-chatbot-api
```

### Verify Deployment

```bash
# Check container status
docker ps

# Check logs
docker logs techgear-chatbot

# Test health endpoint
curl http://localhost:8000/health

# Test API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the price of SmartWatch?"}'
```

### Services & Ports

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Redis**: localhost:6379

---

## Cloud Deployment

### AWS Deployment (ECS/Fargate)

#### 1. Setup AWS CLI

```bash
aws configure
```

#### 2. Create ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name techgear-chatbot

# Get repository URI
ECR_URI=$(aws ecr describe-repositories --repository-names techgear-chatbot --query 'repositories[0].repositoryUri' --output text)
```

#### 3. Build & Push Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI

# Build image
docker build -t techgear-chatbot .

# Tag and push
docker tag techgear-chatbot:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

#### 4. Deploy to ECS

```bash
# Create cluster
aws ecs create-cluster --cluster-name techgear-cluster

# Create task definition (see deployment/aws-task-definition.json)
aws ecs register-task-definition --cli-input-json file://deployment/aws-task-definition.json

# Create service
aws ecs create-service \
  --cluster techgear-cluster \
  --service-name techgear-service \
  --task-definition techgear-chatbot \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### GCP Deployment (Cloud Run)

#### 1. Setup GCP

```bash
gcloud init
gcloud config set project YOUR_PROJECT_ID
```

#### 2. Build & Deploy

```bash
# Enable required APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/techgear-chatbot

# Deploy to Cloud Run
gcloud run deploy techgear-chatbot \
  --image gcr.io/YOUR_PROJECT_ID/techgear-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Azure Deployment (Container Instances)

#### 1. Setup Azure CLI

```bash
az login
az account set --subscription YOUR_SUBSCRIPTION_ID
```

#### 2. Create Resources

```bash
# Create resource group
az group create --name techgear-rg --location eastus

# Create container registry
az acr create --resource-group techgear-rg --name techgearacr --sku Basic

# Build and push
az acr build --registry techgearacr --image techgear-chatbot:latest .

# Deploy container
az container create \
  --resource-group techgear-rg \
  --name techgear-chatbot \
  --image techgearacr.azurecr.io/techgear-chatbot:latest \
  --cpu 2 --memory 4 \
  --dns-name-label techgear-chatbot \
  --ports 8000 \
  --environment-variables GEMINI_API_KEY=$GEMINI_API_KEY
```

---

## Monitoring & Logging

### Prometheus Metrics

Access metrics at: http://localhost:9090

Key metrics:
- API request count
- Response times
- Error rates
- System resources (CPU, memory)

### Grafana Dashboards

Access Grafana at: http://localhost:3000

Default credentials: `admin/admin`

Import dashboards:
1. Go to Dashboards → Import
2. Load pre-configured dashboards from `monitoring/grafana/dashboards/`

### Logging

Logs are available in multiple locations:

```bash
# Docker logs
docker logs techgear-chatbot -f

# Application logs
tail -f logs/app.log

# Structured JSON logs
cat logs/app.log | jq '.'
```

### Alerts

Configure alerting in `monitoring/prometheus.yml`:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts.yml'
```

---

## Security

### HTTPS/TLS

#### Using Let's Encrypt

```bash
# Install certbot
sudo apt-get install certbot

# Generate certificate
sudo certbot certonly --standalone -d yourdomain.com

# Update nginx configuration
cp ssl/cert.pem /etc/letsencrypt/live/yourdomain.com/fullchain.pem
cp ssl/key.pem /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

#### Using Nginx Reverse Proxy

```bash
# Start with nginx
docker-compose --profile with-nginx up -d
```

### API Key Authentication

Enable in `.env`:

```bash
API_KEY_AUTH_ENABLED=true
API_KEYS=key1,key2,key3
```

Use in requests:

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Rate Limiting

Configure in `.env`:

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### Network Security

- Use firewall rules to restrict access
- Enable VPC/Security Groups on cloud platforms
- Use private subnets for databases
- Enable DDoS protection

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker logs techgear-chatbot

# Common causes:
# - Missing .env file
# - Invalid API key
# - Port already in use
```

#### 2. API Returns 500 Errors

```bash
# Check application logs
docker exec techgear-chatbot cat /app/logs/app.log

# Verify environment variables
docker exec techgear-chatbot env | grep GEMINI
```

#### 3. Slow Response Times

```bash
# Check resource usage
docker stats techgear-chatbot

# Scale workers
docker-compose up -d --scale api=3
```

#### 4. ChromaDB Issues

```bash
# Clear and rebuild vector database
rm -rf chroma_db
docker exec techgear-chatbot python src/services/embeddings_service.py
```

### Debug Mode

Enable debug logging:

```bash
# Update .env
LOG_LEVEL=DEBUG

# Restart container
docker-compose restart api
```

---

## Operations

### Backup & Restore

#### Backup

```bash
# Backup script
./deployment/backup.sh

# Manual backup
tar -czf backup-$(date +%Y%m%d).tar.gz \
  chroma_db/ \
  logs/ \
  .env
```

#### Restore

```bash
# Extract backup
tar -xzf backup-20260130.tar.gz

# Restart services
docker-compose restart
```

### Scaling

#### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale api=3

# Load balancer (nginx)
# Update upstream in nginx.conf
```

#### Vertical Scaling

```bash
# Update docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

### Updates & Rollbacks

#### Update

```bash
# Pull latest code
git pull origin main

# Deploy new version
VERSION=v2.0.0 ./deployment/deploy.sh compose
```

#### Rollback

```bash
# Automatic rollback
./deployment/deploy.sh rollback

# Manual rollback
git checkout v1.0.0
docker-compose up -d --build
```

### Maintenance

#### Regular Tasks

- **Daily**: Check logs for errors
- **Weekly**: Review metrics and performance
- **Monthly**: Update dependencies, rotate logs
- **Quarterly**: Security audit, backup verification

#### Health Checks

```bash
# Automated health check
watch -n 30 'curl -s http://localhost:8000/health | jq .'

# Full system check
docker ps
docker stats --no-stream
curl http://localhost:8000/health
```

---

## Support

### Getting Help

- **Documentation**: Check this guide and API docs
- **Issues**: GitHub Issues
- **Email**: support@techgear.com

### Performance Tuning

- Adjust worker count based on CPU cores
- Enable Redis caching for frequent queries
- Use CDN for static assets
- Implement query result caching
- Optimize ChromaDB chunk size

---

## Appendix

### Environment Variables Reference

See `.env.production` for complete list with defaults.

### Port Reference

| Service    | Port | Purpose              |
|------------|------|----------------------|
| API        | 8000 | Main API             |
| Prometheus | 9090 | Metrics collection   |
| Grafana    | 3000 | Dashboards           |
| Redis      | 6379 | Caching              |
| Nginx      | 80   | HTTP                 |
| Nginx      | 443  | HTTPS                |

### Useful Commands

```bash
# View all containers
docker ps -a

# View logs
docker-compose logs -f api

# Shell into container
docker exec -it techgear-chatbot bash

# Restart specific service
docker-compose restart api

# View metrics
curl http://localhost:8001/metrics

# Database shell
docker exec -it techgear-chatbot-redis redis-cli
```

---

**Last Updated**: January 30, 2026  
**Version**: 1.0.0
