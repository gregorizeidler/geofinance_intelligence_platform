# 🎨 Technical Architecture Diagrams

## 💰 Credit Risk Intelligence Flow

Our spatial credit risk assessment process combines traditional financial metrics with advanced geographic intelligence:

```mermaid
flowchart LR
    subgraph "📍 Location Input"
        LOC["🌐 Geographic Point<br/>(Lat, Lon)"]
    end
    
    subgraph "🗺️ Spatial Processing"
        H3_CONV["⬡ H3 Conversion<br/>Point → Hexagon ID"]
        NEIGHBORS["🔗 Neighborhood Analysis<br/>K-ring spatial context"]
        FEATURES["⚙️ Feature Extraction<br/>50+ spatial indicators"]
    end
    
    subgraph "💼 Business Context"
        BUSINESS["🏢 Business Profile<br/>Type, Age, Revenue"]
        FINANCIAL["💰 Financial Data<br/>Credit history, Ratios"]
        COMBINED["🔄 Data Fusion<br/>Spatial + Business features"]
    end
    
    subgraph "🧠 AI Processing"
        MODEL["🤖 XGBoost Model<br/>Spatial intelligence"]
        SHAP["📊 SHAP Explainer<br/>Feature importance"]
        SCORE["🎯 Risk Score<br/>0-1000 scale"]
    end
    
    subgraph "📋 Decision Output"
        CATEGORY["🏷️ Risk Category<br/>Low/Medium/High/Critical"]
        RECOMMENDATION["💡 Decision<br/>Approve/Review/Decline"]
        FACTORS["🔍 Key Factors<br/>Spatial contributors"]
    end
    
    LOC --> H3_CONV
    H3_CONV --> NEIGHBORS
    NEIGHBORS --> FEATURES
    
    BUSINESS --> COMBINED
    FINANCIAL --> COMBINED
    FEATURES --> COMBINED
    
    COMBINED --> MODEL
    MODEL --> SHAP
    MODEL --> SCORE
    
    SCORE --> CATEGORY
    SHAP --> FACTORS
    CATEGORY --> RECOMMENDATION
    FACTORS --> RECOMMENDATION
    
    classDef input fill:#e3f2fd,stroke:#0277bd,stroke-width:3px,color:#000
    classDef spatial fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000
    classDef business fill:#fff8e1,stroke:#f57c00,stroke-width:3px,color:#000
    classDef ai fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
    classDef output fill:#f1f8e9,stroke:#558b2f,stroke-width:3px,color:#000
    
    class LOC input
    class H3_CONV,NEIGHBORS,FEATURES spatial
    class BUSINESS,FINANCIAL,COMBINED business
    class MODEL,SHAP,SCORE ai
    class CATEGORY,RECOMMENDATION,FACTORS output
```

## 🏪 Market Intelligence Pipeline

Our market opportunity analysis leverages multi-dimensional spatial data to identify optimal expansion locations:

```mermaid
graph TD
    subgraph "🎯 Market Analysis Pipeline"
        direction TB
        
        subgraph "📊 Demand Analysis"
            POP["👥 Population Density<br/>Demographics & Income"]
            EDU["🎓 Education Levels<br/>Purchasing power proxy"]
            EMP["💼 Employment Rates<br/>Economic stability"]
        end
        
        subgraph "🏪 Supply Analysis"
            COMPETITORS["🏢 Competitor Density<br/>Market saturation"]
            DIVERSITY["🎭 Business Diversity<br/>Market completeness"]
            MATURITY["📈 Market Maturity<br/>Growth potential"]
        end
        
        subgraph "🚇 Infrastructure"
            TRANSPORT["🚌 Transportation Access<br/>Customer reachability"]
            ROADS["🛣️ Road Connectivity<br/>Logistics efficiency"]
            CENTRALITY["🎯 Urban Centrality<br/>Strategic positioning"]
        end
        
        subgraph "💰 Financial Ecosystem"
            TTV["💳 Transaction Volume<br/>Payment activity"]
            RISK["⚖️ Risk Environment<br/>Business stability"]
            ADOPTION["📱 Digital Adoption<br/>Tech readiness"]
        end
    end
    
    subgraph "🧮 Scoring Engine"
        WEIGHTS["⚖️ Weighted Scoring<br/>Multi-criteria analysis"]
        COMPOSITE["🎯 Opportunity Score<br/>0.0 - 1.0 scale"]
        ROI["💹 ROI Calculation<br/>Expected returns"]
    end
    
    subgraph "🎨 Strategic Output"
        HEATMAP["🌡️ Opportunity Heatmap<br/>Geographic visualization"]
        RANKING["📊 Location Ranking<br/>Priority classification"]
        STRATEGY["📋 Action Plan<br/>Market entry strategy"]
    end
    
    POP --> WEIGHTS
    EDU --> WEIGHTS
    EMP --> WEIGHTS
    
    COMPETITORS --> WEIGHTS
    DIVERSITY --> WEIGHTS
    MATURITY --> WEIGHTS
    
    TRANSPORT --> WEIGHTS
    ROADS --> WEIGHTS
    CENTRALITY --> WEIGHTS
    
    TTV --> WEIGHTS
    RISK --> WEIGHTS
    ADOPTION --> WEIGHTS
    
    WEIGHTS --> COMPOSITE
    COMPOSITE --> ROI
    
    COMPOSITE --> HEATMAP
    ROI --> RANKING
    RANKING --> STRATEGY
    
    classDef demand fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#000
    classDef supply fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000
    classDef infrastructure fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000
    classDef financial fill:#fce4ec,stroke:#e91e63,stroke-width:2px,color:#000
    classDef scoring fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#000
    classDef output fill:#e8f5e8,stroke:#4caf50,stroke-width:2px,color:#000
    
    class POP,EDU,EMP demand
    class COMPETITORS,DIVERSITY,MATURITY supply
    class TRANSPORT,ROADS,CENTRALITY infrastructure
    class TTV,RISK,ADOPTION financial
    class WEIGHTS,COMPOSITE,ROI scoring
    class HEATMAP,RANKING,STRATEGY output
```

## 🗺️ Spatial Data Architecture

Built on Uber's H3 hexagonal grid system for consistent, efficient spatial analysis:

```mermaid
graph TB
    subgraph "🌍 Geographic Coverage"
        direction LR
        POA["🏙️ Porto Alegre<br/>Metropolitan Area"]
        BOUNDS["📐 Spatial Bounds<br/>1,000+ km²"]
    end
    
    subgraph "⬡ H3 Hexagonal Grid"
        direction TB
        RES9["🔍 Resolution 9<br/>~174m edge length"]
        HEXS["⬢ 1,000+ Hexagons<br/>Complete coverage"]
        INDEX["📇 Spatial Indexing<br/>Uber H3 system"]
    end
    
    subgraph "📊 Feature Categories"
        direction TB
        
        subgraph "👥 Socioeconomic"
            INCOME["💰 Average Income"]
            AGE["👶👨👴 Age Distribution"]
            EDUC["🎓 Education Levels"]
            UNEMP["📊 Unemployment Rate"]
        end
        
        subgraph "🏪 Commercial"
            POI_COUNT["🏢 POI Density"]
            MERCHANT["💳 Merchant Count"]
            DIVERSITY["🎭 Business Diversity"]
            COMPETITION["⚔️ Competition Level"]
        end
        
        subgraph "🚇 Infrastructure"
            ROADS["🛣️ Road Density"]
            TRANSPORT["🚌 Transit Access"]
            CENTRAL["🎯 Centrality Score"]
            CONNECT["🔗 Connectivity Index"]
        end
        
        subgraph "💎 Spatial Intelligence"
            NEIGHBORS["🔗 Neighborhood Effects"]
            CLUSTERS["🎯 Spatial Clusters"]
            HOTSPOTS["🔥 Business Hotspots"]
            ISOLATION["🏝️ Isolation Metrics"]
        end
    end
    
    subgraph "⚡ Real-time Processing"
        QUERY["❓ Location Query<br/>(lat, lon)"]
        LOOKUP["🔍 Hexagon Lookup<br/>O(1) complexity"]
        ENRICH["⚙️ Feature Enrichment<br/>Sub-second response"]
        RESULT["📊 Spatial Intelligence<br/>50+ features"]
    end
    
    POA --> RES9
    BOUNDS --> RES9
    RES9 --> HEXS
    HEXS --> INDEX
    
    INDEX --> INCOME
    INDEX --> AGE
    INDEX --> EDUC
    INDEX --> UNEMP
    
    INDEX --> POI_COUNT
    INDEX --> MERCHANT
    INDEX --> DIVERSITY
    INDEX --> COMPETITION
    
    INDEX --> ROADS
    INDEX --> TRANSPORT
    INDEX --> CENTRAL
    INDEX --> CONNECT
    
    INDEX --> NEIGHBORS
    INDEX --> CLUSTERS
    INDEX --> HOTSPOTS
    INDEX --> ISOLATION
    
    QUERY --> LOOKUP
    LOOKUP --> ENRICH
    ENRICH --> RESULT
    
    classDef geo fill:#e1f5fe,stroke:#01579b,stroke-width:3px,color:#000
    classDef h3 fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px,color:#000
    classDef socio fill:#fff8e1,stroke:#f57c00,stroke-width:2px,color:#000
    classDef commercial fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000  
    classDef infra fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000
    classDef spatial fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef processing fill:#ffebee,stroke:#d32f2f,stroke-width:3px,color:#000
    
    class POA,BOUNDS geo
    class RES9,HEXS,INDEX h3
    class INCOME,AGE,EDUC,UNEMP socio
    class POI_COUNT,MERCHANT,DIVERSITY,COMPETITION commercial
    class ROADS,TRANSPORT,CENTRAL,CONNECT infra
    class NEIGHBORS,CLUSTERS,HOTSPOTS,ISOLATION spatial
    class QUERY,LOOKUP,ENRICH,RESULT processing
```

## ☁️ Production Deployment Architecture

Scalable cloud infrastructure for enterprise deployment:

```mermaid
graph TB
    subgraph "☁️ Cloud Infrastructure"
        direction TB
        
        subgraph "🌐 Load Balancer"
            LB["⚖️ Load Balancer<br/>High Availability"]
        end
        
        subgraph "🚀 Application Layer"
            API1["🔌 API Instance 1<br/>FastAPI + Uvicorn"]
            API2["🔌 API Instance 2<br/>FastAPI + Uvicorn"]  
            API3["🔌 API Instance 3<br/>FastAPI + Uvicorn"]
        end
        
        subgraph "💾 Data Layer"
            direction LR
            POSTGRES["🐘 PostgreSQL + PostGIS<br/>Spatial database"]
            REDIS["🔴 Redis Cache<br/>Feature store"]
            S3["🪣 Object Storage<br/>Data lake"]
        end
        
        subgraph "🤖 ML Services"
            direction LR
            MODEL_API["🧠 Model Server<br/>XGBoost inference"]
            FEATURE_API["⚙️ Feature Store<br/>Spatial features"]
            BATCH_PROC["📊 Batch Processing<br/>Feature updates"]
        end
        
        subgraph "📊 Monitoring"
            direction LR
            METRICS["📈 Metrics<br/>Prometheus"]
            LOGS["📝 Logging<br/>ELK Stack"]
            ALERTS["🚨 Alerting<br/>PagerDuty"]
        end
    end
    
    subgraph "🔧 Development"
        direction TB
        DOCKER["🐳 Docker<br/>Containerization"]
        CI_CD["🔄 CI/CD Pipeline<br/>Automated deployment"]
        TESTS["🧪 Test Suite<br/>Quality assurance"]
    end
    
    subgraph "👥 Users"
        FINTECH["🏦 Fintech Apps<br/>REST API clients"]
        ANALYSTS["📊 Data Analysts<br/>Jupyter notebooks"]
        DEVS["👩‍💻 Developers<br/>SDK integration"]
    end
    
    FINTECH --> LB
    ANALYSTS --> LB
    DEVS --> LB
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> POSTGRES
    API2 --> POSTGRES
    API3 --> POSTGRES
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> MODEL_API
    API2 --> MODEL_API
    API3 --> MODEL_API
    
    MODEL_API --> FEATURE_API
    FEATURE_API --> S3
    BATCH_PROC --> POSTGRES
    BATCH_PROC --> S3
    
    API1 --> METRICS
    API2 --> METRICS
    API3 --> METRICS
    
    METRICS --> ALERTS
    LOGS --> ALERTS
    
    DOCKER --> CI_CD
    TESTS --> CI_CD
    CI_CD --> API1
    CI_CD --> API2
    CI_CD --> API3
    
    classDef cloud fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,color:#000
    classDef app fill:#e8f5e8,stroke:#388e3c,stroke-width:3px,color:#000
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:3px,color:#000
    classDef ml fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
    classDef monitoring fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#000
    classDef dev fill:#ffebee,stroke:#d32f2f,stroke-width:3px,color:#000
    classDef users fill:#e0f2f1,stroke:#00695c,stroke-width:3px,color:#000
    
    class LB cloud
    class API1,API2,API3 app
    class POSTGRES,REDIS,S3 data
    class MODEL_API,FEATURE_API,BATCH_PROC ml
    class METRICS,LOGS,ALERTS monitoring
    class DOCKER,CI_CD,TESTS dev
    class FINTECH,ANALYSTS,DEVS users
```