"""Image Generator Agent for FlowForge - supports multiple diagram types."""

import base64
import re
import time
from typing import Any, Optional

import requests
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.prompts import PromptTemplate

from src.agents.base_agent import BaseAgent
from src.config import Config
from src.schemas.request import DiagramType

try:
    from mermaid import Mermaid
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False
    Mermaid = None


class ImageGeneratorAgent(BaseAgent):
    """Generate Mermaid diagrams from project plans."""

    # Diagram types relevant to each audience
    STAKEHOLDER_DIAGRAM_TYPES = {DiagramType.FLOWCHART, DiagramType.GANTT, DiagramType.ARCHITECTURE}

    # Stakeholder-focused templates: business language, no technical jargon
    STAKEHOLDER_DIAGRAM_TEMPLATES = {
        DiagramType.FLOWCHART: {
            "system_role": (
                "You are a business analyst presenting project phases to executive stakeholders. "
                "Use plain business language. FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT include CI/CD pipelines, API gateways, microservices, "
                "infrastructure components, tech stack names, or any engineering implementation detail. "
                "FORBIDDEN: Do NOT add parallel side-by-side branches — keep a single column top-to-bottom. "
                "Every arrow must flow strictly forward (top to bottom). Show phases as a clean linear progression. "
                "MAXIMUM 12 nodes total."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- MAXIMUM 12 nodes total — phases only, no sub-tasks\n"
                "- Show business phases as a STRICTLY LINEAR top-to-bottom flow — single column, no side-by-side branches\n"
                "- Use {} for gate/review decisions, [] for phases — each gate has exactly 2 forward-only outcomes\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: parallel side-by-side branches — they make the diagram unreadably wide\n"
                "- FORBIDDEN: backward edges or loops pointing upward\n"
                "- FORBIDDEN: Docker, Kubernetes, APIs, databases, CI/CD, deployments, or infrastructure terms\n"
                "- Example:\n"
                "graph TD\n"
                "    Discovery[Discovery and Planning] --> DesignGate{Scope Approved?}\n"
                "    DesignGate -->|Yes| Design[Design and Prototype]\n"
                "    DesignGate -->|Needs Work| Refinement[Scope Refinement]\n"
                "    Refinement --> Design\n"
                "    Design --> BuildGate{Design Approved?}\n"
                "    BuildGate -->|Yes| Build[Build and Deliver]\n"
                "    BuildGate -->|Revise| DesignRevision[Design Revision]\n"
                "    DesignRevision --> Build\n"
                "    Build --> UAT[User Acceptance Testing]\n"
                "    UAT --> LaunchGate{Ready to Launch?}\n"
                "    LaunchGate -->|Yes| GoLive[Go Live]\n"
                "    LaunchGate -->|Not Yet| FinalFixes[Final Fixes]\n"
                "    FinalFixes --> GoLive"
            ),
        },
        DiagramType.GANTT: {
            "system_role": (
                "You are a project manager presenting a milestone timeline to business stakeholders. "
                "Show only key milestones and phases. FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT include sprint names, deployment pipelines, infra setup tasks, "
                "or any technical engineering work items — only business-visible milestones and phases."
            ),
            "requirements": (
                "- You MUST start with 'gantt' on the first line\n"
                "- You MUST include 'title' and 'dateFormat YYYY-MM-DD'\n"
                "- Use 'section' for phases; keep task names business-friendly\n"
                "- Mark final deliverables as milestones\n"
                "- FORBIDDEN tasks: CI/CD setup, infrastructure provisioning, technical spike, "
                "docker build, kubernetes config, or any engineering-internal task\n"
                "- Example:\n"
                "gantt\n"
                "    title Project Milestones\n"
                "    dateFormat YYYY-MM-DD\n"
                "    section Initiation\n"
                "    Project Kickoff :a1, 2026-05-14, 5d\n"
                "    section Planning\n"
                "    Business Requirements :a2, after a1, 7d\n"
                "    section Delivery\n"
                "    Build & Test :a3, after a2, 21d\n"
                "    section Launch\n"
                "    Go Live :milestone, after a3, 0d"
            ),
        },
        DiagramType.ARCHITECTURE: {
            "system_role": (
                "You are presenting a Solution Overview to non-technical executive stakeholders. "
                "Structure the diagram as exactly 5 abstract tiers flowing top to bottom: "
                "Tier 1 — People (who uses the system), "
                "Tier 2 — Channels (how they access it, e.g. web portal, mobile app, dashboard), "
                "Tier 3 — Business Capabilities (what the system does, named as business functions), "
                "Tier 4 — Information (what data is captured, stored, or reported), "
                "Tier 5 — Operations (how the solution is run and supported, e.g. Cloud Hosting, Support Team). "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT name any technology products, protocols, or infrastructure tools. "
                "Do NOT include Docker, Kubernetes, Redis, Kafka, PostgreSQL, API Gateway, or any engineering terms. "
                "Every label must be a plain English business phrase a non-technical executive would understand."
            ),
            "requirements": (
                "- Use graph TD syntax with one subgraph per tier — exactly 5 subgraphs\n"
                "- Tier order top to bottom: People → Channels → Business Capabilities → Information → Operations\n"
                "- Each tier has 2–4 nodes representing distinct business roles, touchpoints, capabilities, data areas, or operational functions\n"
                "- Connect tiers with arrows flowing strictly downward — ONLY connect nodes to the ADJACENT tier below\n"
                "- FORBIDDEN: skip-tier connections (e.g. Tier 1 directly to Tier 3) — they cause crossed diagonal lines\n"
                "- FORBIDDEN: backward arrows pointing upward to a previous tier\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase — no spaces in IDs\n"
                "- FORBIDDEN: technology product names, acronyms, infrastructure terms, engineering jargon\n"
                "- Example structure (adapt content to the actual project):\n"
                "graph TD\n"
                "    subgraph T1[People]\n"
                "        FieldStaff[Field Staff]\n"
                "        Managers[Managers]\n"
                "    end\n"
                "    subgraph T2[Channels]\n"
                "        MobileApp[Mobile App]\n"
                "        WebPortal[Web Portal]\n"
                "    end\n"
                "    subgraph T3[Business Capabilities]\n"
                "        Scheduling[Work Scheduling]\n"
                "        Reporting[Performance Reporting]\n"
                "        Alerts[Automated Alerts]\n"
                "    end\n"
                "    subgraph T4[Information]\n"
                "        ActivityRecords[Activity Records]\n"
                "        PerformanceData[Performance Data]\n"
                "    end\n"
                "    subgraph T5[Operations]\n"
                "        CloudHosting[Cloud Hosting]\n"
                "        SupportTeam[Support Team]\n"
                "    end\n"
                "    FieldStaff --> MobileApp\n"
                "    Managers --> WebPortal\n"
                "    MobileApp --> Scheduling\n"
                "    WebPortal --> Reporting\n"
                "    Scheduling --> ActivityRecords\n"
                "    Reporting --> PerformanceData\n"
                "    Alerts --> Managers\n"
                "    ActivityRecords --> CloudHosting\n"
                "    PerformanceData --> CloudHosting\n"
                "    CloudHosting --> SupportTeam"
            ),
        },
    }

    DIAGRAM_TEMPLATES = {
        DiagramType.WORKFLOW: {
            "system_role": (
                "You are an expert technical workflow designer for software engineering teams. "
                "STEP 1 — read the Plan and extract: the exact tech stack, service names, APIs, and team roles. "
                "STEP 2 — build the workflow using ONLY those extracted names. Never invent generic labels. "
                "STEP 3 — UNIQUE NODE IDs AND LABELS: every node must be declared as `uniqueId[Display Label]`. "
                "NEVER write a bare node ID without a bracket label — `backendUnitTests` alone is wrong; "
                "`BeUnitTests[Backend Unit Tests Pass?]` is correct. "
                "If two parallel tracks have a similar step, give each a unique ID and unique label: "
                "MLUnitTests[ML Unit Tests Pass?] and BeUnitTests[Backend Unit Tests Pass?]. "
                "STEP 4 — ALL PARALLEL BRANCHES MUST CONVERGE: every parallel branch must eventually merge back "
                "into a single shared node (e.g. DockerBuild or ProdRelease). No branch may be a dead end. "
                "STEP 5 — KEEP IT SMALL: maximum 15 nodes total. Combine minor steps into one node. "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: vague labels like 'Service', 'Module', 'Component', 'Phase 1', or any business-only language."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- MAXIMUM 15 nodes total — combine minor steps, keep only critical gates and handoffs\n"
                "- Show the end-to-end technical development workflow: from requirements through CI/CD to production\n"
                "- Include team handoffs and build/test gates — only the most critical ones\n"
                "- Use {} for decision gates, [] for process steps — every node must reference a real service, team, or technical action\n"
                "- Maximum 2 parallel branches at any point — both MUST converge within 3 steps to a shared node\n"
                "- CONVERGENCE: every parallel branch MUST eventually connect to a shared downstream node — no dead-end boxes\n"
                "- UNIQUE IDs: if two tracks have similar steps (e.g. unit tests), give them distinct IDs like mlUnitTests and beUnitTests\n"
                "- EVERY node must use the form `nodeId[Display Label]` or `nodeId{Decision Label}` — bare IDs are invalid\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase — globally unique across the whole diagram\n"
                "- FORBIDDEN: bare node IDs without a display label (e.g. writing `backendUnitTests` alone on a line)\n"
                "- FORBIDDEN: more than 15 nodes — trim ruthlessly\n"
                "- FORBIDDEN: generic labels without specific context from the plan\n"
                "- Example pattern (adapt every label to the actual project):\n"
                "graph TD\n"
                "    Req[Requirements Finalised] --> DesignReview{Architecture Review}\n"
                "    DesignReview -->|Approved| MLModelDev[YOLO Model Training]\n"
                "    DesignReview -->|Approved| BackendDev[FastAPI Backend Dev]\n"
                "    DesignReview -->|Revise| Req\n"
                "    MLModelDev --> MLUnitTests{ML Unit Tests Pass?}\n"
                "    MLUnitTests -->|Yes| MLOpsReg[MLflow Model Registry]\n"
                "    MLUnitTests -->|No| MLModelDev\n"
                "    BackendDev --> BeUnitTests{Backend Unit Tests Pass?}\n"
                "    BeUnitTests -->|Yes| BeIntegTest[FastAPI Integration Tests]\n"
                "    BeUnitTests -->|No| BackendDev\n"
                "    MLOpsReg --> DockerBuild[Docker Image Build]\n"
                "    BeIntegTest --> DockerBuild\n"
                "    DockerBuild --> K8sDeploy[Kubernetes Deploy]\n"
                "    K8sDeploy --> SmokeTest{Smoke Tests Pass?}\n"
                "    SmokeTest -->|Yes| ProdRelease[Production Release]\n"
                "    SmokeTest -->|No| DockerBuild"
            ),
        },
        DiagramType.CI_CD: {
            "system_role": (
                "You are an expert DevOps/MLOps engineer designing a CI/CD pipeline. "
                "STEP 1 — read the Plan and extract the exact CI/CD tools, container registry, orchestration platform, "
                "cloud provider, and any ML model deployment steps mentioned. "
                "STEP 2 — build the pipeline using those exact tool names. "
                "STEP 3 — SINGLE CONNECTED GRAPH: the entire pipeline MUST be one connected graph with a single "
                "entry point (the code push node). There must be NO disconnected sub-flows or floating node groups. "
                "Every node must be reachable by following edges from the entry point. "
                "If the project uses a separate ML model deploy step (e.g. GCR Push → Cloud Run deploy), "
                "that step must branch OFF an existing node in the main pipeline — not start a new disconnected graph. "
                "STEP 4 — GATES: include an 'All Tests Pass?' gate after the test stages with a 'No' path to Rollback. "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: generic stage names without tool context — every stage must reference a real tool or service."
            ),
            "requirements": (
                "- Use graph TD syntax — ONE connected graph, single entry point\n"
                "- Show the full pipeline: code push → lint → test (unit + integration + model validation) → "
                "gate → build → staging deploy → approval gate → production deploy → rollback path\n"
                "- GATE: after all test stages, include an {All Tests Pass?} decision: Yes → build, No → Rollback\n"
                "- If the project has an ML component, branch the model registry push and inference deploy OFF "
                "the Docker Build node — they must be CONNECTED to the main flow, not a separate island\n"
                "- Name the actual tools: GitHub Actions / GitLab CI, Docker, ECR/GCR/DockerHub, "
                "Kubernetes/ECS, ArgoCD/Helm, MLflow/BentoML/TorchServe — whichever the plan specifies\n"
                "- Show rollback as a separate node reachable from the test gate AND the approval gate\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: disconnected sub-flows — every node must connect to the main pipeline\n"
                "- FORBIDDEN: business language, vague stage names — use real DevOps tool names from the plan\n"
                "- Example pattern (adapt tools and stages to the actual project):\n"
                "graph TD\n"
                "    Push[Git Push to main] --> GHActions[GitHub Actions Trigger]\n"
                "    GHActions --> Lint[Ruff Lint + Type Check]\n"
                "    Lint --> UnitTest[PyTest Unit Tests]\n"
                "    UnitTest --> ModelTest[Model Accuracy Validation]\n"
                "    ModelTest --> TestGate{All Tests Pass?}\n"
                "    TestGate -->|Yes| DockerBuild[Docker Build + GCR Push]\n"
                "    TestGate -->|No| Rollback[Rollback + Notify]\n"
                "    DockerBuild --> MLflowReg[MLflow Model Registry Push]\n"
                "    DockerBuild --> StagingDeploy[Helm Deploy to Staging]\n"
                "    MLflowReg --> StagingDeploy\n"
                "    StagingDeploy --> IntTest[Integration Tests on Staging]\n"
                "    IntTest --> ApprovalGate{Manual Approval}\n"
                "    ApprovalGate -->|Approved| ProdDeploy[Kubernetes Deploy to Production]\n"
                "    ApprovalGate -->|Rejected| Rollback\n"
                "    ProdDeploy --> HealthCheck{Health Check}\n"
                "    HealthCheck -->|Pass| Done[Pipeline Complete]\n"
                "    HealthCheck -->|Fail| Rollback"
            ),
        },
        DiagramType.SYSTEM_DESIGN: {
            "system_role": (
                "You are a senior system architect producing a system design diagram. "
                "STEP 1 — read the Plan carefully and extract: the key services, databases, queues, caches, "
                "ML models, APIs, and external integrations. Pick the MOST IMPORTANT ones only. "
                "STEP 2 — draw those components as nodes (8-12 total). Show data flows between them "
                "with labeled arrows indicating the protocol or data type (e.g. REST, gRPC, RTSP, WebSocket, SQL). "
                "STEP 3 — LAYOUT RULES to prevent arrows crossing boxes: "
                "(a) only connect nodes to their ADJACENT subgraph — never skip a subgraph layer; "
                "(b) no node should have more than 3 outgoing edges; "
                "(c) arrange subgraphs so data flows strictly top-to-bottom with no upward arrows. "
                "STEP 4 — FORCE VERTICAL ORDERING: immediately after all subgraph definitions, write a "
                "vertical spine chain connecting one representative node from each subgraph to the next. "
                "This chain is REQUIRED — it tells the layout engine the correct top-to-bottom order. "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: generic placeholders like 'Service', 'Database', 'Backend', 'Queue' — "
                "every node must use the actual technology or service name from the plan."
            ),
            "requirements": (
                "- Use graph TD syntax with subgraph to group components by domain boundary\n"
                "- Subgraphs ordered top-to-bottom: Clients → Ingestion → API Layer → ML Inference → Data Storage → Observability\n"
                "  (use only the subgraphs relevant to the project, but keep this top-to-bottom order)\n"
                "- MAXIMUM 12 nodes total across all subgraphs — choose only the most critical components\n"
                "- SPINE: your very first edges after all subgraph definitions must be a chain: "
                "one node per subgraph connected in order top-to-bottom. This forces the layout engine to stack them correctly.\n"
                "- LAYOUT: connect nodes only to their adjacent subgraph — no skip-layer edges\n"
                "- LAYOUT: max 3 outgoing arrows per node\n"
                "- Label every arrow with the protocol or data format: REST, gRPC, RTSP, SQL, Pub/Sub, WebSocket, etc.\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: any node named 'Service', 'Backend', 'Database', 'Storage', 'Model' without a qualifier\n"
                "- FORBIDDEN: arrows that skip subgraph layers (they cause boxes to overlap)\n"
                "- Example pattern — the vertical spine chain appears first, then all detailed edges:\n"
                "graph TD\n"
                "    subgraph Clients[Clients]\n"
                "        ControlRoom[Control Room Dashboard]\n"
                "        MobileApp[Mobile Alert App]\n"
                "    end\n"
                "    subgraph Ingestion[Video Ingestion]\n"
                "        RTSPStream[RTSP Camera Stream]\n"
                "        FrameExtractor[OpenCV Frame Extractor]\n"
                "        KafkaTopic[Kafka video-frames Topic]\n"
                "    end\n"
                "    subgraph APILayer[API Layer]\n"
                "        FastAPI[FastAPI Gateway]\n"
                "        AlertSvc[Alert Service]\n"
                "    end\n"
                "    subgraph MLLayer[ML Inference]\n"
                "        TorchServe[TorchServe Model Server]\n"
                "        YOLOModel[YOLO Inference Worker]\n"
                "        MLflowReg[MLflow Model Registry]\n"
                "    end\n"
                "    subgraph DataLayer[Data Storage]\n"
                "        PostgreSQL[PostgreSQL Events DB]\n"
                "        Redis[Redis Alert Cache]\n"
                "        S3[S3 Frame Archive]\n"
                "    end\n"
                "    subgraph Observability[Observability]\n"
                "        Prometheus[Prometheus]\n"
                "        Grafana[Grafana Dashboard]\n"
                "    end\n"
                "    ControlRoom -->|RTSP| RTSPStream\n"
                "    RTSPStream -->|frames| FastAPI\n"
                "    FastAPI -->|consume| TorchServe\n"
                "    TorchServe -->|SQL| PostgreSQL\n"
                "    PostgreSQL -->|metrics| Prometheus\n"
                "    RTSPStream -->|RTSP| FrameExtractor\n"
                "    FrameExtractor -->|Publish| KafkaTopic\n"
                "    KafkaTopic -->|Consume| TorchServe\n"
                "    MLflowReg -->|Load Model| TorchServe\n"
                "    TorchServe -->|Run| YOLOModel\n"
                "    YOLOModel -->|REST| FastAPI\n"
                "    FastAPI -->|Alert| AlertSvc\n"
                "    FastAPI -->|Cache| Redis\n"
                "    FrameExtractor -->|Archive| S3\n"
                "    AlertSvc -->|WebSocket| ControlRoom\n"
                "    AlertSvc -->|Push| MobileApp\n"
                "    YOLOModel -->|SQL| PostgreSQL\n"
                "    Prometheus -->|Visualise| Grafana"
            ),
        },
        DiagramType.FLOWCHART: {
            "system_role": (
                "You are an expert technical process analyst mapping the end-to-end engineering process. "
                "STEP 1 — read the Plan and extract the actual technical teams, tools, services, and handoff points. "
                "STEP 2 — build the flowchart showing how work moves through those real technical components. "
                "STEP 3 — STRICTLY VERTICAL NARROW LAYOUT: the diagram must flow top-to-bottom as a single column. "
                "NO parallel branches — use a strict linear sequence with decision gates only. "
                "A decision gate has exactly 2 outcomes: one continues forward, one loops back or goes to a fix step. "
                "STEP 4 — ONE TERMINAL: the diagram must end at exactly one final node (Production Release). "
                "Every path must eventually reach that terminal. "
                "STEP 5 — MAXIMUM 14 NODES TOTAL. Count every node — if you exceed 14, remove the least critical ones. "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: parallel side-by-side branches — every step must be directly above or below another."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- MAXIMUM 14 nodes total — trim to only the most critical technical steps and gates\n"
                "- STRICTLY LINEAR: no side-by-side parallel tracks — one column top-to-bottom only\n"
                "- Include 2-3 decision gates that reflect actual technical conditions (e.g. 'Tests Pass?', 'Accuracy OK?')\n"
                "- Each decision gate has exactly 2 outputs: one forward (Yes/Pass) and one backward loop (No/Fail)\n"
                "- ONE TERMINAL: exactly one final node (Production Release or equivalent)\n"
                "- ALL DECISION PATHS: every {decision} node must have both outputs connected — no dangling edges\n"
                "- Use {} for technical decision gates, [] for technical process steps\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase — unique across the diagram\n"
                "- FORBIDDEN: any parallel tracks or side-by-side branches — they make the diagram unreadably wide\n"
                "- FORBIDDEN: more than 14 nodes total\n"
                "- FORBIDDEN: business-only labels, generic phase names without technical specificity\n"
                "- Example pattern (adapt to actual project — strict single-column flow):\n"
                "graph TD\n"
                "    DataCollection[Dataset Collection] --> Preprocessing[OpenCV Preprocessing]\n"
                "    Preprocessing --> ModelTrain[YOLO Model Training]\n"
                "    ModelTrain --> AccuracyGate{mAP above threshold?}\n"
                "    AccuracyGate -->|Yes| ModelRegistry[MLflow Model Registry]\n"
                "    AccuracyGate -->|No| Preprocessing\n"
                "    ModelRegistry --> TorchServe[TorchServe Deploy]\n"
                "    TorchServe --> IntegrationTest[FastAPI Integration Tests]\n"
                "    IntegrationTest --> TestGate{All Tests Pass?}\n"
                "    TestGate -->|Yes| DockerBuild[Docker Image Build]\n"
                "    TestGate -->|No| TorchServe\n"
                "    DockerBuild --> StagingDeploy[Staging K8s Deploy]\n"
                "    StagingDeploy --> SmokeGate{Smoke Tests Pass?}\n"
                "    SmokeGate -->|Yes| ProdRelease[Production Release]\n"
                "    SmokeGate -->|No| DockerBuild"
            ),
        },
        DiagramType.ARCHITECTURE: {
            "system_role": (
                "You are a senior enterprise architect producing a deployment architecture diagram. "
                "STEP 1 — read the Plan and extract: cloud provider, compute services, container orchestration, "
                "databases, caches, message brokers, ML serving infrastructure, monitoring stack, and networking. "
                "STEP 2 — draw EVERY extracted infrastructure component in the correct layer. "
                "ALL nodes must be declared INSIDE a subgraph — no floating nodes outside subgraphs. "
                "STEP 3 — LAYOUT RULES to prevent arrows crossing boxes: "
                "(a) only connect nodes to their ADJACENT layer — never skip a layer; "
                "(b) if two distant layers must connect, add an intermediate relay node between them; "
                "(c) no node should have more than 3 outgoing edges; "
                "(d) data flows strictly top-to-bottom with no upward arrows. "
                "STEP 4 — FORCE VERTICAL ORDERING: immediately after all subgraph definitions, write a "
                "vertical spine chain connecting one representative node from each layer to the next "
                "(e.g. ClientNode --> GatewayNode --> AppNode --> MLNode --> DataNode --> ObsNode). "
                "This spine is REQUIRED — it tells the layout engine the correct top-to-bottom order. "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: vague layer names without real technology names — every node must be a real infra component."
            ),
            "requirements": (
                "- Use graph TD syntax with subgraph for each infrastructure layer\n"
                "- Layers top-to-bottom: Client Tier → API Gateway → Application Services → ML Platform → "
                "Data Tier → Observability — use only layers relevant to the project, keep this order\n"
                "- ALL NODES must be inside a subgraph — no nodes declared outside subgraph blocks\n"
                "- SPINE: your very first edges after all subgraph definitions must be a top-to-bottom chain "
                "connecting one node per layer in order. This forces the layout engine to stack layers correctly.\n"
                "- LAYOUT: connect nodes only to their adjacent layer — no skip-layer edges\n"
                "- LAYOUT: if a node in layer A needs to reach layer C (skipping B), "
                "route it through a relay node in B\n"
                "- LAYOUT: max 3 outgoing arrows per node — split into multiple nodes if needed\n"
                "- Name every component with its actual technology (e.g. NGINX, FastAPI, Kubernetes, "
                "PostgreSQL, Redis, Prometheus, Grafana, MLflow, TorchServe)\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- Include 8-10 nodes — do not add nodes just to reach a count; quality over quantity\n"
                "- FORBIDDEN: layer or node labels without a real technology name\n"
                "- FORBIDDEN: arrows that skip layers (they cause boxes to overlap in the rendered diagram)\n"
                "- FORBIDDEN: nodes declared outside a subgraph block\n"
                "- Example pattern — spine chain appears first to force vertical stacking, then detailed edges:\n"
                "graph TD\n"
                "    subgraph ClientTier[Client Tier]\n"
                "        ReactDash[React Dashboard]\n"
                "        MobileAlert[Mobile Alert Client]\n"
                "    end\n"
                "    subgraph APIGateway[API Gateway]\n"
                "        NGINX[NGINX Ingress]\n"
                "        FastAPI[FastAPI Service]\n"
                "    end\n"
                "    subgraph AppServices[Application Services]\n"
                "        AlertSvc[Alert Microservice]\n"
                "        FrameIngest[Frame Ingestion Service]\n"
                "        KafkaBroker[Kafka Broker]\n"
                "    end\n"
                "    subgraph MLPlatform[ML Platform]\n"
                "        TorchServe[TorchServe Server]\n"
                "        MLflow[MLflow Registry]\n"
                "    end\n"
                "    subgraph DataTier[Data Tier]\n"
                "        PostgreSQL[PostgreSQL]\n"
                "        Redis[Redis Cache]\n"
                "        S3[S3 Storage]\n"
                "    end\n"
                "    subgraph Observability[Observability]\n"
                "        MetricsRelay[Metrics Relay]\n"
                "        Prometheus[Prometheus]\n"
                "        Grafana[Grafana Dashboard]\n"
                "    end\n"
                "    ReactDash -->|HTTPS| NGINX\n"
                "    NGINX -->|HTTP| AlertSvc\n"
                "    AlertSvc -->|Publish| KafkaBroker\n"
                "    KafkaBroker -->|Consume| TorchServe\n"
                "    TorchServe -->|SQL| PostgreSQL\n"
                "    PostgreSQL -->|Metrics| MetricsRelay\n"
                "    MobileAlert -->|HTTPS| NGINX\n"
                "    NGINX -->|HTTP| FastAPI\n"
                "    FastAPI -->|RPC| FrameIngest\n"
                "    FrameIngest -->|Publish| KafkaBroker\n"
                "    TorchServe -->|Load| MLflow\n"
                "    AlertSvc -->|Cache| Redis\n"
                "    FrameIngest -->|Upload| S3\n"
                "    AppServices -->|Scrape| MetricsRelay\n"
                "    MLPlatform -->|Scrape| MetricsRelay\n"
                "    MetricsRelay -->|Forward| Prometheus\n"
                "    Prometheus -->|Visualise| Grafana"
            ),
        },
        DiagramType.GANTT: {
            "system_role": (
                "You are a senior engineering project manager creating a detailed sprint-level delivery timeline. "
                "STEP 1 — read the Plan and extract: all engineering TEAMS (e.g. ML Team, Backend Team, DevOps, QA, Frontend), "
                "their specific technical deliverables, dependencies, and the total timeline in weeks. "
                "STEP 2 — break the work into concrete engineering tasks PER TEAM. "
                "Each section header MUST be a team name, NOT a phase name. "
                "FORBIDDEN section names: 'Discovery', 'Design', 'Development', 'Deployment', 'Planning' — "
                "these are phases, not teams. Use 'ML Team', 'Backend Team', 'Frontend Team', 'DevOps Team', 'QA' instead. "
                "FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: high-level milestone-only timelines — this is for engineers who need task-level detail."
            ),
            "requirements": (
                "- You MUST start with 'gantt' on the first line — no other syntax\n"
                "- You MUST include 'title' and 'dateFormat YYYY-MM-DD'\n"
                "- Sections MUST be team names: ML Team, Backend Team, Frontend Team, DevOps Team, QA\n"
                "- FORBIDDEN section names: Discovery, Design, Development, Deployment, Planning, Testing — use team names only\n"
                "- Each task must be a specific engineering deliverable owned by that team\n"
                "- Use 'after' dependencies to show the critical path\n"
                "- Mark key releases and model deployments as milestones\n"
                "- Include 8–12 tasks across 3–4 sections — keep task names SHORT (under 40 chars each)\n"
                "- FORBIDDEN: tasks named 'Development', 'Testing', 'Deployment' without a qualifier — "
                "use names like 'FastAPI endpoint dev', 'YOLO model training', 'K8s cluster setup'\n"
                "- Example pattern (adapt task names, durations, and team names to the actual project):\n"
                "gantt\n"
                "    title Engineering Delivery Timeline\n"
                "    dateFormat YYYY-MM-DD\n"
                "    section ML Team\n"
                "    Dataset preparation :ml1, 2026-05-19, 7d\n"
                "    YOLO model training :ml2, after ml1, 14d\n"
                "    Model evaluation and tuning :ml3, after ml2, 7d\n"
                "    MLflow registry push :milestone, after ml3, 0d\n"
                "    section Backend Team\n"
                "    FastAPI project scaffold :be1, 2026-05-19, 3d\n"
                "    Video ingestion service :be2, after be1, 7d\n"
                "    Alert and event API :be3, after be2, 7d\n"
                "    PostgreSQL schema and migrations :be4, after be1, 5d\n"
                "    section DevOps Team\n"
                "    K8s cluster provisioning :do1, 2026-05-19, 5d\n"
                "    CI/CD pipeline setup :do2, after do1, 5d\n"
                "    TorchServe deployment config :do3, after ml3, 5d\n"
                "    section QA\n"
                "    Integration test suite :qa1, after be3, 7d\n"
                "    Load and latency testing :qa2, after qa1, 5d\n"
                "    Production release :milestone, after qa2, 0d"
            ),
        },
    }

    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize image generator agent."""
        super().__init__("image_agent", session_manager=session_manager)

        self.llm: Optional[InferenceClient] = None

        self.logger.info("Initializing ImageGeneratorAgent.")

        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        self.logger.info("Initializing image generation LLM.")

        if not Config.HF_TOKEN:
            self.logger.error("HF_TOKEN missing in configuration.")
            raise ValueError("HF_TOKEN must be set.")

        self.llm = InferenceClient(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=Config.HF_TOKEN,
        )

        self.logger.info(
            "Image generator LLM initialized successfully with model=%s",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        )

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from LLM response."""
        self.logger.info(
            "Extracting text from response type=%s",
            type(response).__name__,
        )

        if response is None:
            self.logger.warning("Received empty response object.")
            return ""

        if isinstance(response, str):
            self.logger.info(
                "Response is raw string with length=%d",
                len(response),
            )
            return response

        extracted = getattr(response, "content", str(response))

        self.logger.info(
            "Extracted response text length=%d",
            len(extracted),
        )

        return extracted

    def _build_diagram_prompt(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
        audience_type: str = "engineer",
    ) -> str:
        """Build prompt for diagram generation."""
        self.logger.info(
            "Building prompt for diagram_type=%s audience=%s",
            diagram_type.value,
            audience_type,
        )

        self.logger.info(
            "Plan length=%d | Timeline length=%d",
            len(plan),
            len(timeline) if timeline else 0,
        )

        if audience_type == "stakeholder":
            template_config = self.STAKEHOLDER_DIAGRAM_TEMPLATES.get(
                diagram_type,
                self.STAKEHOLDER_DIAGRAM_TEMPLATES[DiagramType.FLOWCHART],
            )
        else:
            template_config = self.DIAGRAM_TEMPLATES.get(
                diagram_type,
                self.DIAGRAM_TEMPLATES[DiagramType.FLOWCHART],
            )

        prompt_template = PromptTemplate(
    input_variables=[
        "system_role",
        "rules",
        "plan",
        "timeline",
        "diagram_type_name",
    ],
    template="""
{system_role}

STRICT RULES — follow exactly:
{rules}

Use the plan below to fill in real project-specific content (services, phases, tasks).
Do NOT copy the example verbatim — adapt it to the actual project.

Plan:
{plan}

Timeline Reference:
{timeline}

Output ONLY the Mermaid diagram. No markdown fences. No explanation. No text after the last line.
""",
)

        formatted_prompt = prompt_template.format(
            system_role=template_config["system_role"],
            rules=template_config["requirements"],
            plan=plan,
            timeline=timeline or "No timeline provided",
            diagram_type_name=diagram_type.value,
        )

        self.logger.info(
            "Prompt built successfully | length=%d",
            len(formatted_prompt),
        )

        self.logger.info(
            "Prompt preview:\n%s",
            formatted_prompt[:1000],
        )

        return formatted_prompt

    def _parse_diagram(self, response: str) -> str:
        """Extract Mermaid diagram from response."""
        self.logger.info("Parsing Mermaid diagram from LLM response.")

        diagram = self._extract_text(response).strip()

        self.logger.info("Raw diagram length=%d", len(diagram))

        # Strip DeepSeek-R1 <think>...</think> reasoning blocks
        diagram = re.sub(r"<think>.*?</think>", "", diagram, flags=re.DOTALL).strip()

        # Extract from ```mermaid ... ``` fences if present
        if "```" in diagram:
            self.logger.warning("Markdown fences detected. Cleaning response.")
            # Try to extract just the content inside the first ```mermaid...``` block
            mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", diagram, re.DOTALL | re.IGNORECASE)
            if mermaid_match:
                diagram = mermaid_match.group(1).strip()
            else:
                diagram = (
                    diagram.replace("```mermaid", "")
                    .replace("```", "")
                    .strip()
                )

        # If there's preamble before the diagram keyword, strip it
        valid_starts = ["gantt", "flowchart", "graph", "sequenceDiagram",
                        "classDiagram", "stateDiagram", "erDiagram", "pie", "journey", "gitGraph"]
        lines = diagram.split("\n")
        for i, line in enumerate(lines):
            if any(line.strip().lower().startswith(v.lower()) for v in valid_starts):
                diagram = "\n".join(lines[i:]).strip()
                break

        # Strip trailing prose that LLMs append after the diagram
        diagram = self._strip_trailing_prose(diagram)

        # Sanitize unquoted multi-word node IDs (e.g. "Discovery --> Requirement Analysis" → "Discovery --> RequirementAnalysis")
        diagram = self._sanitize_node_ids(diagram)

        self.logger.info("Parsed diagram length=%d", len(diagram))
        self.logger.info("Diagram preview:\n%s", diagram[:1000])

        return diagram

    def _sanitize_node_ids(self, diagram: str) -> str:
        """Fix common LLM-generated Mermaid syntax errors.
        
        - Removes parentheses inside [] node labels (breaks Mermaid parser)
        - Collapses unquoted multi-word node IDs after arrows
        """
        lines = diagram.split("\n")
        sanitized = []
        
        for line in lines:
            # Remove parentheses inside [] node labels: A[Label (detail)] → A[Label detail]
            line = re.sub(r'\[([^\]]*)\(([^\)]*)\)([^\]]*)\]', r'[\1\2\3]', line)
            
            # Skip further processing if line uses brackets/braces (labels already safe)
            if "[" in line or "{" in line:
                sanitized.append(line)
                continue
            
            # Skip lines with () node labels (stadium shape) — those are intentional
            if "(" in line:
                sanitized.append(line)
                continue
            
            # Fix unquoted multi-word node IDs after arrows
            line = re.sub(
                r'(-->|->)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                lambda m: f"{m.group(1)} {m.group(2).replace(' ', '')}",
                line
            )
            line = re.sub(
                r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(-->|->)',
                lambda m: f"{m.group(1).replace(' ', '')} {m.group(2)}",
                line
            )
            
            sanitized.append(line)
        
        return "\n".join(sanitized)

    def _strip_trailing_prose(self, diagram: str) -> str:
        """Remove explanatory prose appended after the Mermaid diagram."""
        lines = diagram.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped and any(
                stripped.startswith(prefix)
                for prefix in ["This ", "The ", "Note:", "Here ", "Above ", "Below "]
            ):
                self.logger.info("Stripping trailing prose: %s", stripped[:60])
                break
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _build_fallback_gantt(self, timeline: Optional[str] = None) -> str:
        """Build a minimal valid gantt diagram when the LLM returns wrong syntax."""
        import datetime
        start = datetime.date.today().strftime("%Y-%m-%d")
        return (
            f"gantt\n"
            f"    title Project Timeline\n"
            f"    dateFormat YYYY-MM-DD\n"
            f"    section Discovery\n"
            f"    Requirements Analysis :a1, {start}, 5d\n"
            f"    section Design\n"
            f"    System Design :a2, after a1, 7d\n"
            f"    section Development\n"
            f"    Backend Development :a3, after a2, 14d\n"
            f"    Frontend Development :a4, after a2, 14d\n"
            f"    section Testing\n"
            f"    QA Testing :a5, after a3, 7d\n"
            f"    section Deployment\n"
            f"    Release :milestone, after a5, 0d"
        )

    def _validate_diagram(self, diagram: str) -> bool:
        """Validate Mermaid syntax structure."""
        self.logger.info("Validating generated Mermaid diagram.")

        if not diagram or len(diagram.strip()) < 10:
            self.logger.warning(
                "Diagram too short or empty."
            )
            return False

        valid_starts = [
            "gantt",
            "flowchart",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
            "graph",
        ]

        diagram_lower = diagram.lower().strip()

        self.logger.info(
            "Diagram starts with: %s",
            diagram_lower[:50],
        )

        is_valid = any(
            diagram_lower.startswith(v.lower())
            for v in valid_starts
        )

        self.logger.info(
            "Diagram validation result=%s",
            is_valid,
        )

        return is_valid

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check whether error is retryable."""
        self.logger.warning(
            "Checking retryable status for error=%s",
            exc,
        )

        status_code = None

        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)

            if response:
                status_code = getattr(response, "status_code", None)

            if status_code is None:
                status_code = getattr(exc, "status_code", None)

            if status_code is None and exc.args:
                match = re.search(r"(\d{3})\s", str(exc.args[0]))
                if match:
                    status_code = int(match.group(1))

            self.logger.warning(
                "HF HTTP status code=%s",
                status_code,
            )

            return status_code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            if response:
                status_code = getattr(response, "status_code", None)
            self.logger.warning(
                "Requests HTTP status code=%s",
                status_code,
            )
            return status_code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

        retryable = isinstance(
            exc,
            (TimeoutError, ConnectionError),
        )

        self.logger.warning(
            "Retryable generic exception=%s",
            retryable,
        )

        return retryable

    def _chat_completion_with_retry(
        self,
        messages: list,
        max_tokens: int,
    ) -> Any:
        """Execute LLM request with retries."""
        last_exception = None

        max_attempts = 3
        base_delay = 1.0

        self.logger.info(
            "Starting diagram LLM request | max_tokens=%d",
            max_tokens,
        )

        self.logger.info(
            "Message preview:\n%s",
            messages[0]["content"][:1000],
        )

        for attempt in range(max_attempts):
            try:
                self.logger.info(
                    "LLM attempt %d/%d",
                    attempt + 1,
                    max_attempts,
                )

                start_time = time.time()

                response = self.llm.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                )

                elapsed = round(time.time() - start_time, 2)

                self.logger.info(
                    "LLM request succeeded in %.2fs",
                    elapsed,
                )

                if hasattr(response, "choices"):
                    content = response.choices[0].message.content

                    self.logger.info(
                        "Response content length=%d",
                        len(content) if content else 0,
                    )

                    self.logger.info(
                        "Response preview:\n%s",
                        content[:1000] if content else "EMPTY",
                    )

                return response

            except Exception as exc:
                last_exception = exc

                self.logger.error(
                    "LLM request failed on attempt %d/%d | error=%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    exc_info=True,
                )

                if not self._is_retryable_error(exc):
                    self.logger.error(
                        "Encountered non-retryable error."
                    )
                    raise exc

                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)

                    self.logger.warning(
                        "Retrying in %.1f seconds.",
                        delay,
                    )

                    time.sleep(delay)

        self.logger.error(
            "All LLM attempts failed."
        )

        raise last_exception

    def generate_diagram(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
        audience_type: str = "engineer",
    ) -> dict[str, Any]:
        """Generate a Mermaid diagram."""
        try:
            self.logger.info(
                "Generating diagram type=%s",
                diagram_type.value,
            )

            if not plan:
                self.logger.error(
                    "Plan missing. Cannot generate diagram."
                )

                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": None,
                    "is_valid": False,
                    "error": "No plan provided",
                    "image_data": None,
                }

            self.logger.info(
                "Plan length=%d",
                len(plan),
            )

            prompt = self._build_diagram_prompt(
                plan,
                diagram_type,
                timeline,
                audience_type=audience_type,
            )

            # Gantt diagrams need more tokens for multi-section task lists;
            # graph diagrams must stay small to avoid mermaid.ink URL-length limits.
            token_budget = 1200 if diagram_type == DiagramType.GANTT else 900

            response = self._chat_completion_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=token_budget,
            )

            msg = response.choices[0].message if hasattr(response, "choices") else None
            raw_content = (
                (msg and msg.content and msg.content.strip())
                or (getattr(msg, "reasoning_content", None) or "").strip()
                or (getattr(msg, "reasoning", None) or "").strip()
                or ""
            ) if msg else ""

            self.logger.info(
                "Raw LLM content length=%d",
                len(raw_content) if raw_content else 0,
            )

            diagram = self._parse_diagram(raw_content)

            # If gantt was requested but LLM returned wrong type, build a minimal valid gantt
            if diagram_type == DiagramType.GANTT and not diagram.lower().strip().startswith("gantt"):
                self.logger.warning(
                    "Gantt diagram requested but LLM returned wrong type. Using fallback gantt."
                )
                diagram = self._build_fallback_gantt(timeline)

            is_valid = self._validate_diagram(diagram)

            if not is_valid:
                self.logger.warning(
                    "Generated Mermaid diagram failed validation."
                )

                self.logger.warning(
                    "Invalid Mermaid output:\n%s",
                    diagram,
                )

                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": diagram,
                    "is_valid": False,
                    "error": "No valid Mermaid diagram generated",
                    "image_data": None,
                }

            self.logger.info(
                "Diagram validated successfully."
            )

            image_data = None

            if MERMAID_AVAILABLE:
                # mermaid.ink encodes the diagram as base64 in the URL.
                # Large diagrams (>2000 chars) may cause 414/503 from mermaid.ink.
                if len(diagram) > 2000:
                    self.logger.warning(
                        "Diagram code is %d chars — may exceed mermaid.ink URL limit.",
                        len(diagram),
                    )

                max_render_attempts = 3
                render_delay = 1.0
                for render_attempt in range(max_render_attempts):
                    try:
                        self.logger.info(
                            "Attempting Mermaid PNG rendering (attempt %d/%d).",
                            render_attempt + 1,
                            max_render_attempts,
                        )
                        mm = Mermaid(diagram, width=1200, scale=1.5)
                        resp = mm.img_response

                        # Validate HTTP response before treating as image bytes.
                        status = getattr(resp, "status_code", None)
                        if status is not None and status != 200:
                            raise RuntimeError(
                                f"mermaid.ink returned HTTP {status} "
                                f"(diagram may be too long or has syntax errors)"
                            )

                        image_bytes = resp.content

                        # Reject obviously non-image responses (empty or HTML error pages).
                        # Accept PNG (\x89PNG), JPEG (\xff\xd8\xff), and other binary formats.
                        if not image_bytes or image_bytes[:1] in (b"<", b"{"):
                            raise RuntimeError(
                                f"mermaid.ink response is not an image "
                                f"(got {len(image_bytes)} bytes, "
                                f"starts with {image_bytes[:16]!r})"
                            )

                        # Detect format for the data URI (PNG vs JPEG)
                        fmt = "jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "png"
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        image_data = f"data:image/{fmt};base64,{image_base64}"
                        self.logger.info(
                            "Diagram image rendered successfully | format=%s | size=%d bytes.",
                            fmt,
                            len(image_bytes),
                        )
                        break

                    except Exception as exc:
                        self.logger.warning(
                            "Failed Mermaid rendering (attempt %d/%d) | error=%s",
                            render_attempt + 1,
                            max_render_attempts,
                            exc,
                        )
                        if render_attempt < max_render_attempts - 1:
                            time.sleep(render_delay)
                            render_delay *= 2

                if image_data is None:
                    self.logger.error(
                        "All %d render attempts failed for diagram_type=%s | "
                        "diagram length=%d chars. Diagram will be missing from PDF.",
                        max_render_attempts,
                        diagram_type.value,
                        len(diagram),
                    )
            else:
                self.logger.warning("Mermaid renderer unavailable.")

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": diagram,
                "title": diagram_type.value.replace("_", " ").title(),
                "description": (
                    f"Auto-generated {diagram_type.value} diagram"
                ),
                "is_valid": True,
                "validation_feedback": "Diagram generated successfully",
                "error": None,
                "image_data": image_data,
            }

        except Exception as exc:
            self.logger.error(
                "Diagram generation failed for type=%s | error=%s",
                diagram_type.value,
                exc,
                exc_info=True,
            )

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": None,
                "is_valid": False,
                "error": (
                    f"Failed to generate {diagram_type.value} "
                    f"diagram: {exc}"
                ),
                "image_data": None,
            }

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate all requested diagrams."""
        try:
            self.logger.info(
                "Starting image generation workflow."
            )

            plan = state.get("plan")
            timeline = state.get("timetable")
            audience_type = state.get("audience_type", "engineer")

            improvement_prompt = state.get("improvement_prompt")
            if improvement_prompt:
                self.logger.info("Using improvement prompt for regeneration")
                plan = f"{plan}\n\n--- IMPROVEMENT FEEDBACK ---\n{improvement_prompt}"

            diagram_types_raw = state.get(
                "diagram_types",
                [DiagramType.WORKFLOW],
            )

            self.logger.info(
                "Plan exists=%s | timeline exists=%s",
                bool(plan),
                bool(timeline),
            )

            self.logger.info(
                "Requested diagram types=%s",
                diagram_types_raw,
            )

            if not plan:
                self.logger.error(
                    "No plan provided to image generator."
                )

                return self._update_state(
                    state,
                    {
                        "error": (
                            "Failed to generate images: "
                            "no plan provided"
                        ),
                        "diagrams": [],
                        "current_agent": "image_agent",
                    },
                )

            # Parse diagram types
            if isinstance(diagram_types_raw, list):
                if all(
                    isinstance(d, str)
                    for d in diagram_types_raw
                ):
                    diagram_types = [
                        DiagramType(d)
                        for d in diagram_types_raw
                    ]
                else:
                    diagram_types = diagram_types_raw

            elif isinstance(diagram_types_raw, str):
                diagram_types = [
                    DiagramType(diagram_types_raw)
                ]

            else:
                diagram_types = [DiagramType.WORKFLOW]

            self.logger.info(
                "Resolved %d diagram types.",
                len(diagram_types),
            )

            # Stakeholders only receive business-friendly diagram types
            if audience_type == "stakeholder":
                diagram_types = [
                    dt for dt in diagram_types
                    if dt in self.STAKEHOLDER_DIAGRAM_TYPES
                ]
                if not diagram_types:
                    diagram_types = [DiagramType.FLOWCHART, DiagramType.GANTT]
                self.logger.info(
                    "Stakeholder audience: filtered to %d diagram types.",
                    len(diagram_types),
                )

            diagrams = []

            previous_invalid = self._get_previous_invalid_output("diagrams")
            if previous_invalid:
                self.logger.info(
                    "Found previous invalid output for reference"
                )

            for idx, diagram_type in enumerate(diagram_types):
                self.logger.info(
                    "Generating diagram %d/%d | type=%s",
                    idx + 1,
                    len(diagram_types),
                    diagram_type.value,
                )

                diagram_result = self.generate_diagram(
                    plan=plan,
                    diagram_type=diagram_type,
                    timeline=timeline,
                    audience_type=audience_type,
                )

                self.logger.info(
                    "Diagram generation result | valid=%s | error=%s",
                    diagram_result.get("is_valid"),
                    diagram_result.get("error"),
                )

                diagrams.append(diagram_result)

            valid_count = sum(
                1 for d in diagrams if d.get("is_valid")
            )

            self.logger.info(
                "Generated %d/%d valid diagrams.",
                valid_count,
                len(diagrams),
            )

            for idx, diagram in enumerate(diagrams):
                self.logger.info(
                    "Diagram #%d summary | type=%s | valid=%s",
                    idx + 1,
                    diagram.get("diagram_type"),
                    diagram.get("is_valid"),
                )

            result_state = self._update_state(
                state,
                {
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                    "error": (
                        None
                        if valid_count > 0
                        else "No valid diagrams generated"
                    ),
                    "current_agent": "image_agent",
                },
            )

            self._save_session_output(
                output_type="diagrams",
                output_data={
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                },
                feedback=result_state.get("error"),
                is_valid=valid_count > 0,
            )

            return result_state

        except Exception as exc:
            self.logger.error(
                "Image generator workflow failed | error=%s",
                exc,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": (
                        f"Image generator agent failed: {exc}"
                    ),
                    "diagrams": [],
                    "diagram_count": 0,
                    "valid_diagram_count": 0,
                    "current_agent": "image_agent",
                },
            )