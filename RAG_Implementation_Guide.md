# RAG (检索增强生成) 完整实现说明文档

## 目录
1. [系统架构概述](#系统架构概述)
2. [核心模块说明](#核心模块说明)
3. [数据流程详解](#数据流程详解)
4. [数据库设计](#数据库设计)
5. [前端实现](#前端实现)
6. [后端实现](#后端实现)
7. [向量数据库集成](#向量数据库集成)
8. [配置指南](#配置指南)
9. [API接口说明](#api接口说明)
10. [部署与使用](#部署与使用)

---

## 系统架构概述

本项目实现了一套完整的RAG系统，支持多种文档类型的知识库构建、向量化存储和智能检索。

### 技术栈
- **后端**: Java + Spring Boot + MyBatis
- **前端**: Vue3 + Ant Design Vue + TypeScript
- **向量数据库**: Milvus / Weaviate (可配置切换)
- **嵌入模型**: OpenAI / Ollama / 智谱AI / SiliconFlow / 阿里百炼
- **数据库**: MySQL
- **缓存**: Redis

### 架构图
```
┌─────────────────────────────────────────────────────────────────┐
│                           用户界面层                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ 知识库管理 │  │ 文件上传   │  │ 角色权限   │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          API 接口层                              │
│  KnowledgeController / KnowledgeRoleController                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         业务逻辑层                               │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │ 知识库服务      │  │ 向量库服务      │                      │
│  │ - 创建/删除     │  │ - 存储向量      │                      │
│  │ - 文件上传      │  │ - 相似度搜索    │                      │
│  │ - 内容管理      │  │ - 删除向量      │                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       文档处理层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ 文档加载器  │  │ 文本分割器  │  │ 嵌入模型    │            │
│  │ - Word      │  │ - 字符级    │  │ - OpenAI    │            │
│  │ - PDF       │  │ - 自定义    │  │ - Ollama    │            │
│  │ - Excel     │  │ - 重叠分块  │  │ - 智谱AI    │            │
│  │ - Text      │  └─────────────┘  │ - 阿里百炼  │            │
│  └─────────────┘                   └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         数据存储层                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │ MySQL      │  │ 向量数据库 │  │ Redis      │               │
│  │ - 知识库   │  │ - Milvus   │  │ - 缓存     │               │
│  │ - 附件     │  │ - Weaviate │  │ - 验证码   │               │
│  │ - 片段     │  └────────────┘  └────────────┘               │
│  └────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块说明

### 1. 知识库管理模块 (Knowledge Management)

#### 目录结构
```
backend/ruoyi-ai/
├── ruoyi-modules-api/ruoyi-knowledge-api/
│   └── src/main/java/org/ruoyi/
│       ├── domain/                      # 数据模型
│       │   ├── KnowledgeInfo.java       # 知识库实体
│       │   ├── KnowledgeFragment.java   # 知识片段实体
│       │   └── KnowledgeAttach.java     # 附件实体
│       ├── service/                     # 服务接口
│       │   ├── IKnowledgeInfoService.java
│       │   ├── IKnowledgeFragmentService.java
│       │   └── IKnowledgeAttachService.java
│       └── mapper/                      # 数据库映射
│           ├── KnowledgeInfoMapper.java
│           ├── KnowledgeFragmentMapper.java
│           └── KnowledgeAttachMapper.java
│
└── ruoyi-modules/ruoyi-chat/
    └── src/main/java/org/ruoyi/chat/
        ├── controller/knowledge/
        │   └── KnowledgeController.java # RESTful API
        └── service/knowledge/
            └── KnowledgeInfoServiceImpl.java # 业务实现
```

#### 核心功能
- 知识库创建与配置
- 文件上传与解析
- 文档分块与向量化
- 知识片段管理
- 权限控制

---

### 2. 文档处理模块 (Document Processing)

#### 文档加载器 (ResourceLoader)
**位置**: `ruoyi-knowledge-api/src/main/java/org/ruoyi/chain/loader/`

支持的文档类型:
```java
// 文档类型枚举
public enum DocType {
    WORD,    // .doc, .docx
    PDF,     // .pdf
    EXCEL,   // .xls, .xlsx
    TXT,     // .txt
    MD,      // .md
    HTML     // .html
}
```

**核心类**: `ResourceLoaderFactory.java`
```java
public ResourceLoader getLoaderByFileType(DocType docType) {
    return switch (docType) {
        case WORD -> new WordLoader();
        case PDF -> new PdfLoader();
        case EXCEL -> new ExcelLoader();
        case TXT -> new TextLoader();
        default -> throw new IllegalArgumentException("Unsupported file type");
    };
}
```

#### 文本分割器 (TextSplitter)
**位置**: `ruoyi-knowledge-api/src/main/java/org/ruoyi/chain/split/`

**核心类**: `CharacterTextSplitter.java`

分块策略:
```java
public List<String> split(String content, String kid) {
    // 获取知识库配置
    KnowledgeInfoVo config = knowledgeInfoService.queryById(kid);

    String separator = config.getKnowledgeSeparator();     // 分隔符
    int blockSize = config.getTextBlockSize();             // 块大小
    int overlapSize = config.getOverlapChar();             // 重叠大小

    // 分块逻辑
    if (content.contains(separator)) {
        // 按自定义分隔符分割
        return Arrays.asList(content.split(separator));
    } else {
        // 按固定大小分割，支持重叠
        return splitBySize(content, blockSize, overlapSize);
    }
}
```

**分块示例**:
```
原文本: "ABCDEFGHIJKLMNOPQRSTUVWXYZ" (26个字符)
配置: blockSize=10, overlap=2

分块结果:
Chunk 1: "ABCDEFGHIJ"   (0-9)
Chunk 2: "IJKLMNOPQR"   (8-17, 重叠2字符)
Chunk 3: "QRSTUVWXYZ"   (16-25, 重叠2字符)
```

---

### 3. 向量嵌入模块 (Embedding)

#### 嵌入模型工厂
**位置**: `ruoyi-knowledge-api/src/main/java/org/ruoyi/embedding/EmbeddingModelFactory.java`

支持的提供商:
```java
@Service
public class EmbeddingModelFactory {

    // 根据配置创建嵌入模型
    public BaseEmbedModelService createModel(String modelName, Integer dimension) {
        ChatModelVo config = chatModelService.selectModelByName(modelName);

        return switch (config.getProviderName()) {
            case "openai" -> new OpenAiEmbeddingProvider(config);
            case "ollama" -> new OllamaEmbeddingProvider(config);
            case "zhipuai" -> new ZhiPuAiEmbeddingProvider(config);
            case "siliconflow" -> new SiliconFlowEmbeddingProvider(config);
            case "alibailain" -> new AliBaiLianEmbeddingProvider(config);
            default -> throw new IllegalArgumentException("Unsupported provider");
        };
    }
}
```

#### 嵌入模型提供商实现

**OpenAI 嵌入**
```java
@Component("openai")
public class OpenAiEmbeddingProvider implements BaseEmbedModelService {

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
        OpenAiEmbeddingModel model = OpenAiEmbeddingModel.builder()
            .baseUrl(config.getApiHost())
            .apiKey(config.getApiKey())
            .modelName(config.getModelName())    // text-embedding-3-small
            .dimensions(config.getDimension())   // 1536
            .build();

        return model.embedAll(segments);
    }
}
```

**模型参数配置**:
```
提供商        | 模型名称                    | 维度  | 特点
-------------|----------------------------|------|------------------
OpenAI       | text-embedding-3-small     | 1536 | 高质量，成本低
OpenAI       | text-embedding-3-large     | 3072 | 最高质量
Ollama       | bge-m3                     | 1024 | 本地部署，免费
智谱AI       | embedding-2                | 1024 | 中文优化
SiliconFlow  | BAAI/bge-large-zh-v1.5     | 1024 | 中文，开源
阿里百炼     | text-embedding-v2          | 1536 | 多模态支持
```

---

### 4. 向量存储模块 (Vector Store)

#### 向量库策略模式
**位置**: `ruoyi-knowledge-api/src/main/java/org/ruoyi/service/strategy/`

**抽象策略接口**:
```java
public interface VectorStoreStrategy {
    void createSchema(String kid, String embeddingModelName);
    void storeEmbeddings(StoreEmbeddingBo bo);
    List<String> getQueryVector(QueryVectorBo bo);
    void removeByDocId(String docId, String kid);
}
```

**策略工厂**:
```java
@Component
public class VectorStoreStrategyFactory {

    @Value("${vector-store.type}")
    private String vectorStoreType;  // milvus 或 weaviate

    public VectorStoreStrategy getStrategy() {
        return switch (vectorStoreType.toLowerCase()) {
            case "milvus" -> milvusVectorStoreStrategy;
            case "weaviate" -> weaviateVectorStoreStrategy;
            default -> throw new IllegalArgumentException("Unknown vector store type");
        };
    }
}
```

#### Milvus 实现详解
**位置**: `ruoyi-knowledge-api/src/main/java/org/ruoyi/service/strategy/impl/MilvusVectorStoreStrategy.java`

**1. 创建集合 (Collection)**
```java
public void createSchema(String kid, String embeddingModelName) {
    String collectionName = "knowledge_" + kid;
    int dimension = getModelDimension(embeddingModelName);

    MilvusEmbeddingStore store = MilvusEmbeddingStore.builder()
        .uri(milvusUrl)
        .collectionName(collectionName)
        .dimension(dimension)
        .indexType(IndexType.IVF_FLAT)     // 索引类型
        .metricType(MetricType.L2)         // 距离度量
        .autoFlushOnInsert(true)           // 自动刷新
        .build();
}
```

**Milvus 索引类型对比**:
```
索引类型     | 适用场景           | 查询速度 | 内存占用 | 精确度
------------|-------------------|---------|---------|--------
FLAT        | 小数据集(<1万)     | 慢      | 低      | 100%
IVF_FLAT    | 中等数据集(1-100万) | 中      | 中      | >95%
IVF_SQ8     | 大数据集(>100万)   | 快      | 低      | >90%
HNSW        | 需要极速查询       | 极快    | 高      | >99%
```

**2. 向量存储**
```java
public void storeEmbeddings(StoreEmbeddingBo bo) {
    // 获取嵌入模型
    EmbeddingModel embeddingModel = embeddingModelFactory.createModel(
        bo.getEmbeddingModelName(),
        bo.getDimension()
    );

    // 获取向量存储
    EmbeddingStore<TextSegment> store = getMilvusStore(
        "knowledge_" + bo.getKid(),
        bo.getDimension(),
        false
    );

    // 批量向量化并存储
    for (int i = 0; i < bo.getChunkList().size(); i++) {
        String chunk = bo.getChunkList().get(i);

        // 生成向量
        Embedding embedding = embeddingModel.embed(chunk).content();

        // 创建文本段（带元数据）
        Map<String, Object> metadata = Map.of(
            "kid", bo.getKid(),
            "docId", bo.getDocId(),
            "fid", bo.getFidList().get(i),
            "idx", i
        );
        TextSegment segment = TextSegment.from(chunk, Metadata.from(metadata));

        // 存储到Milvus
        store.add(embedding, segment);
    }
}
```

**3. 相似度搜索**
```java
public List<String> getQueryVector(QueryVectorBo bo) {
    // 向量化查询文本
    Embedding queryEmbedding = embeddingModel.embed(bo.getQuery()).content();

    // 构建搜索请求
    EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
        .queryEmbedding(queryEmbedding)
        .maxResults(bo.getMaxResults())           // Top K
        .minScore(0.7)                            // 相似度阈值
        .build();

    // 执行搜索
    EmbeddingSearchResult<TextSegment> result = embeddingStore.search(request);

    // 提取结果
    List<String> results = new ArrayList<>();
    for (EmbeddingMatch<TextSegment> match : result.matches()) {
        results.add(match.embedded().text());
        // 相似度分数: match.score()
    }

    return results;
}
```

**4. 删除向量**
```java
public void removeByDocId(String docId, String kid) {
    // 构建过滤条件
    Filter filter = MetadataFilterBuilder
        .metadataKey("docId")
        .isEqualTo(docId);

    // 删除匹配的向量
    embeddingStore.removeAll(filter);
}
```

---

### 5. 工作流集成模块 (Workflow Integration)

#### 知识库检索节点
**位置**: `ruoyi-workflow-api/src/main/java/org/ruoyi/workflow/workflow/node/knowledgeRetrieval/KnowledgeRetrievalNode.java`

这是RAG最核心的应用场景，将知识库检索集成到工作流中。

**节点配置结构**:
```java
public class KnowledgeRetrievalNodeConfig {
    private String knowledgeId;           // 知识库ID
    private Integer topK;                 // 返回Top K结果
    private Double similarityThreshold;   // 相似度阈值
    private String retrievalMode;         // 检索模式: vector/graph/hybrid
    private String embeddingModel;        // 嵌入模型
    private Boolean returnSource;         // 是否返回来源信息
    private String queryRewritePrompt;    // 查询改写提示词
}
```

**完整处理流程**:
```java
public NodeProcessResult onProcess() {
    // 1. 获取用户查询
    String userQuery = getFirstInputText();

    // 2. 查询改写 (可选，使用LLM优化查询)
    String optimizedQuery = userQuery;
    if (StringUtils.isNotBlank(config.getQueryRewritePrompt())) {
        optimizedQuery = rewriteQueryWithLLM(userQuery, config.getQueryRewritePrompt());
    }

    // 3. 执行向量检索
    List<String> retrievedChunks = retrieveFromVectorStore(
        config.getKnowledgeId(),
        optimizedQuery,
        config.getTopK(),
        config.getEmbeddingModel()
    );

    // 4. 过滤低相似度结果
    List<String> filteredChunks = filterBySimilarity(
        retrievedChunks,
        config.getSimilarityThreshold()
    );

    // 5. 格式化输出
    String output = String.join("\n\n---\n\n", filteredChunks);

    // 6. 返回结果给下一个节点
    return NodeProcessResult.builder()
        .content(output)
        .metadata(Map.of("sourceCount", filteredChunks.size()))
        .build();
}
```

**查询改写示例**:
```
原始查询: "如何使用RAG?"

改写后查询: "检索增强生成(RAG)的实现方法和使用步骤是什么?"

效果: 改写后的查询更加明确，能检索到更相关的文档片段
```

---

## 数据流程详解

### 文档上传到向量化流程

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 用户上传文档                                               │
│    POST /knowledge/attach/upload                             │
│    - 文件: document.pdf                                      │
│    - 知识库ID: kid_12345                                     │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 文档加载 (ResourceLoader)                                 │
│    - 检测文件类型: PDF                                       │
│    - 使用PdfLoader解析                                       │
│    - 提取纯文本内容                                          │
│    结果: "这是一份关于RAG技术的文档..."                      │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. 保存附件记录 (knowledge_attach表)                         │
│    INSERT INTO knowledge_attach VALUES (                     │
│      id: 1001,                                               │
│      kid: 'kid_12345',                                       │
│      doc_id: 'doc_abc123',                                   │
│      doc_name: 'document.pdf',                               │
│      doc_type: 'pdf',                                        │
│      content: '完整文档内容...',                             │
│      vector_status: 20  -- 处理中                            │
│    )                                                         │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. 文本分块 (CharacterTextSplitter)                          │
│    - 块大小: 500字符                                         │
│    - 重叠: 50字符                                            │
│    - 分隔符: "\n\n"                                          │
│    结果:                                                     │
│      Chunk 1: "这是一份关于RAG技术的文档... [500字符]"       │
│      Chunk 2: "...RAG技术的文档，包括原理... [500字符]"      │
│      Chunk 3: "...包括原理、实现和应用... [500字符]"         │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. 保存知识片段 (knowledge_fragment表)                       │
│    INSERT INTO knowledge_fragment VALUES                     │
│      (id: 2001, kid: 'kid_12345', doc_id: 'doc_abc123',     │
│       fid: 'frag_001', idx: 0, content: 'Chunk 1内容'),     │
│      (id: 2002, kid: 'kid_12345', doc_id: 'doc_abc123',     │
│       fid: 'frag_002', idx: 1, content: 'Chunk 2内容'),     │
│      (id: 2003, kid: 'kid_12345', doc_id: 'doc_abc123',     │
│       fid: 'frag_003', idx: 2, content: 'Chunk 3内容')      │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. 向量化 (Embedding Model)                                  │
│    使用模型: text-embedding-3-small (OpenAI)                 │
│    维度: 1536                                                │
│                                                              │
│    Chunk 1 → [0.123, -0.456, 0.789, ..., 0.234]  (1536维)  │
│    Chunk 2 → [-0.234, 0.567, -0.890, ..., 0.345] (1536维)  │
│    Chunk 3 → [0.345, -0.678, 0.901, ..., -0.456] (1536维)  │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. 存储到向量数据库 (Milvus)                                 │
│    集合名: knowledge_kid_12345                               │
│    索引类型: IVF_FLAT                                        │
│                                                              │
│    Entity 1: {                                               │
│      id: 'frag_001',                                         │
│      vector: [0.123, -0.456, ...],                          │
│      metadata: {kid: 'kid_12345', docId: 'doc_abc123'}      │
│    }                                                         │
│    ... (Entity 2, Entity 3)                                 │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 8. 更新处理状态                                              │
│    UPDATE knowledge_attach                                   │
│    SET vector_status = 30  -- 完成                           │
│    WHERE doc_id = 'doc_abc123'                               │
└──────────────────────────────────────────────────────────────┘
```

### 知识检索流程

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 用户提交查询                                              │
│    查询: "RAG的核心组件有哪些?"                              │
│    知识库ID: kid_12345                                       │
│    Top K: 3                                                  │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 查询改写 (可选，使用LLM)                                  │
│    提示词: "请将用户问题改写为更适合检索的查询"               │
│    原查询: "RAG的核心组件有哪些?"                            │
│    改写后: "检索增强生成系统的主要技术组成部分和模块"         │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. 查询向量化                                                │
│    使用模型: text-embedding-3-small                          │
│    查询文本: "检索增强生成系统的主要技术组成部分和模块"       │
│    结果向量: [0.456, -0.234, 0.678, ..., 0.123] (1536维)    │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. Milvus 向量相似度搜索                                     │
│    集合: knowledge_kid_12345                                 │
│    查询向量: [0.456, -0.234, 0.678, ...]                    │
│    距离度量: L2                                              │
│    Top K: 3                                                  │
│                                                              │
│    搜索结果:                                                 │
│      Match 1: {score: 0.92, fid: 'frag_045'}                │
│      Match 2: {score: 0.88, fid: 'frag_023'}                │
│      Match 3: {score: 0.85, fid: 'frag_067'}                │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. 提取文本内容                                              │
│    从 TextSegment 中获取:                                    │
│                                                              │
│    Result 1: "RAG系统主要包含三个核心组件: 检索器..."       │
│    Result 2: "检索增强生成由文档处理、向量存储..."           │
│    Result 3: "RAG的关键技术包括嵌入模型、向量数据库..."      │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. 过滤和排序                                                │
│    相似度阈值: 0.80                                          │
│    过滤后: Match 1, 2, 3 (都满足)                            │
│                                                              │
│    合并结果:                                                 │
│    "RAG系统主要包含三个核心组件: 检索器...                   │
│     ---                                                      │
│     检索增强生成由文档处理、向量存储...                      │
│     ---                                                      │
│     RAG的关键技术包括嵌入模型、向量数据库..."                │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. 返回给用户或传递给LLM                                     │
│    工作流模式: 传递给下一个LLM节点生成最终答案               │
│    API模式: 直接返回检索结果                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 数据库设计

### ER图
```
┌──────────────────┐         ┌──────────────────┐
│  knowledge_info  │1      N│ knowledge_attach │
│ (知识库)         ├─────────┤ (附件)           │
│  - id (PK)       │         │  - id (PK)       │
│  - kid (UK)      │         │  - kid (FK)      │
│  - kname         │         │  - doc_id (UK)   │
│  - embedding...  │         │  - doc_name      │
└──────────────────┘         │  - content       │
                             └──────────────────┘
                                     │1
                                     │
                                     │N
                             ┌──────────────────┐
                             │knowledge_fragment│
                             │ (知识片段)       │
                             │  - id (PK)       │
                             │  - kid (FK)      │
                             │  - doc_id (FK)   │
                             │  - fid (UK)      │
                             │  - idx           │
                             │  - content       │
                             └──────────────────┘
```

### 表结构详解

#### 1. knowledge_info (知识库表)
```sql
CREATE TABLE `knowledge_info` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `kid` VARCHAR(50) NOT NULL COMMENT '知识库唯一ID',
  `uid` BIGINT(20) COMMENT '所有者用户ID',
  `kname` VARCHAR(50) COMMENT '知识库名称',
  `share` TINYINT(4) DEFAULT 0 COMMENT '是否公开 (0否 1是)',
  `description` VARCHAR(1000) COMMENT '知识库描述',

  -- 分块配置
  `knowledge_separator` VARCHAR(255) DEFAULT '\n\n' COMMENT '文档分隔符',
  `question_separator` VARCHAR(255) COMMENT '问题分隔符',
  `overlap_char` INT DEFAULT 50 COMMENT '重叠字符数',
  `text_block_size` INT DEFAULT 500 COMMENT '文本块大小',

  -- 检索配置
  `retrieve_limit` INT DEFAULT 5 COMMENT '默认检索条数',
  `system_prompt` VARCHAR(255) COMMENT '系统提示词',

  -- 向量化配置
  `vector_model_name` VARCHAR(50) DEFAULT 'milvus' COMMENT '向量库类型',
  `embedding_model_name` VARCHAR(50) COMMENT '嵌入模型名称',
  `embedding_model_id` BIGINT COMMENT '嵌入模型ID',

  -- 审计字段
  `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `create_by` VARCHAR(64),
  `update_by` VARCHAR(64),
  `remark` VARCHAR(500),
  `del_flag` CHAR(1) DEFAULT '0' COMMENT '删除标志 (0正常 1删除)',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_kid` (`kid`),
  KEY `idx_uid` (`uid`),
  KEY `idx_kname` (`kname`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库信息表';
```

**字段说明**:
- `kid`: 知识库全局唯一标识，用于Milvus集合命名
- `text_block_size`: 控制每个chunk的字符数，影响检索粒度
- `overlap_char`: 相邻chunk的重叠字符数，避免上下文断裂
- `retrieve_limit`: 默认返回的相似片段数量
- `embedding_model_name`: 必须与向量化时使用的模型一致

#### 2. knowledge_attach (附件表)
```sql
CREATE TABLE `knowledge_attach` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `kid` VARCHAR(50) NOT NULL COMMENT '知识库ID',
  `doc_id` VARCHAR(50) NOT NULL COMMENT '文档唯一ID',
  `doc_name` VARCHAR(500) COMMENT '文档名称',
  `doc_type` VARCHAR(50) COMMENT '文档类型 (pdf/word/excel/txt)',

  -- 处理状态
  `pic_status` TINYINT DEFAULT 10 COMMENT '图片拆解状态 (10未开始 20进行中 30完成)',
  `pic_anys_status` TINYINT DEFAULT 10 COMMENT '图片分析状态',
  `vector_status` TINYINT DEFAULT 10 COMMENT '向量化状态 (10未开始 20进行中 30完成)',

  -- 内容存储
  `content` LONGTEXT COMMENT '完整文档内容',

  -- 审计
  `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `create_by` VARCHAR(64),
  `update_by` VARCHAR(64),
  `del_flag` CHAR(1) DEFAULT '0',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_doc_id` (`doc_id`),
  UNIQUE KEY `uk_kid_docname` (`kid`, `doc_name`),
  KEY `idx_kid` (`kid`),
  FOREIGN KEY (`kid`) REFERENCES `knowledge_info`(`kid`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库附件表';
```

**状态说明**:
```
10 - 未开始
20 - 处理中
30 - 已完成
40 - 失败
```

#### 3. knowledge_fragment (知识片段表)
```sql
CREATE TABLE `knowledge_fragment` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `kid` VARCHAR(50) NOT NULL COMMENT '知识库ID',
  `doc_id` VARCHAR(50) NOT NULL COMMENT '文档ID',
  `fid` VARCHAR(50) NOT NULL COMMENT '片段ID (向量库中的ID)',
  `idx` INT COMMENT '片段在文档中的索引',
  `content` TEXT COMMENT '片段文本内容',

  `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `del_flag` CHAR(1) DEFAULT '0',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_fid` (`fid`),
  KEY `idx_kid` (`kid`),
  KEY `idx_doc_id` (`doc_id`),
  KEY `idx_kid_docid` (`kid`, `doc_id`),
  FOREIGN KEY (`kid`) REFERENCES `knowledge_info`(`kid`) ON DELETE CASCADE,
  FOREIGN KEY (`doc_id`) REFERENCES `knowledge_attach`(`doc_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识片段表';
```

**关联关系**:
- `fid` 与 Milvus 中的实体ID一一对应
- `idx` 用于保持片段在原文档中的顺序
- 通过 `kid` + `doc_id` 可查询某文档的所有片段

#### 4. knowledge_role (知识库角色表)
```sql
CREATE TABLE `knowledge_role` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `group_id` BIGINT COMMENT '角色组ID',
  `role_name` VARCHAR(50) COMMENT '角色名称',
  `role_desc` VARCHAR(500) COMMENT '角色描述',
  `create_time` DATETIME,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库角色表';
```

#### 5. knowledge_role_relation (知识库角色关联表)
```sql
CREATE TABLE `knowledge_role_relation` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `knowledge_id` BIGINT COMMENT '知识库ID',
  `knowledge_role_id` BIGINT COMMENT '知识库角色ID',
  `create_time` DATETIME,
  PRIMARY KEY (`id`),
  KEY `idx_knowledge_id` (`knowledge_id`),
  KEY `idx_role_id` (`knowledge_role_id`),
  FOREIGN KEY (`knowledge_id`) REFERENCES `knowledge_info`(`id`),
  FOREIGN KEY (`knowledge_role_id`) REFERENCES `knowledge_role`(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库角色关联表';
```

---

## 前端实现

### 目录结构
```
frontend/ruoyi-admin/apps/web-antd/src/
├── api/operator/
│   ├── knowledgeBase/
│   │   └── index.ts                    # 知识库API
│   └── knowledgeRole/
│       ├── knowledge-role.ts           # 角色API
│       └── knowledge-role-group.ts     # 角色组API
│
└── views/operator/
    ├── knowledgeBase/
    │   └── index.vue                   # 知识库管理界面
    └── knowledgeRole/
        ├── knowledgeRole/
        │   ├── index.vue               # 角色管理
        │   └── knowledge-role-drawer.vue
        └── knowledgeRoleGroup/
            ├── index.vue
            └── knowledge-role-group-modal.vue
```

### API封装
**文件**: `src/api/operator/knowledgeBase/index.ts`

```typescript
import { requestClient } from '#/api/request';

// API端点枚举
enum Api {
  knowledgeList = '/knowledge/list',
  knowledgeDelete = '/knowledge/remove',
  knowledgeSave = '/knowledge/save',
  knowledgeDetail = '/knowledge/detail',
  knowledgeFileDelete = '/knowledge/attach/remove',
  knowledgeFragmentList = '/knowledge/fragment/list',
  knowledgeUpload = '/knowledge/attach/upload',
}

// 获取知识库列表
export function knowledgeList() {
  return requestClient.get<KnowledgeListResponse>(Api.knowledgeList);
}

// 创建或更新知识库
export function knowledgeSave(data: KnowledgeSaveRequest) {
  return requestClient.post<void>(Api.knowledgeSave, data);
}

// 获取知识库详情(包含附件和片段)
export function knowledgeDetail(id: string | number) {
  return requestClient.get<KnowledgeDetailResponse>(
    `${Api.knowledgeDetail}/${id}`
  );
}

// 删除知识库
export function knowledgeDelete(kid: string) {
  return requestClient.post<void>(`${Api.knowledgeDelete}/${kid}`);
}

// 删除附件(同时删除向量)
export function knowledgeFileDelete(id: string | number) {
  return requestClient.post<void>(`${Api.knowledgeFileDelete}/${id}`);
}

// 获取知识片段列表
export function knowledgeFragmentList(params: FragmentQueryParams) {
  return requestClient.get<FragmentListResponse>(
    Api.knowledgeFragmentList,
    { params }
  );
}

// TypeScript类型定义
interface KnowledgeSaveRequest {
  kid?: string;                   // 有则更新，无则创建
  kname: string;                  // 知识库名称
  description?: string;           // 描述
  share?: number;                 // 是否公开
  textBlockSize?: number;         // 块大小
  overlapChar?: number;           // 重叠字符
  retrieveLimit?: number;         // 检索条数
  embeddingModelName?: string;    // 嵌入模型
}

interface KnowledgeDetailResponse {
  id: number;
  kid: string;
  kname: string;
  attachments: AttachmentInfo[];
  fragments: FragmentInfo[];
}
```

### 知识库管理界面
**文件**: `src/views/operator/knowledgeBase/index.vue`

```vue
<template>
  <div class="knowledge-base-container">
    <!-- 工具栏 -->
    <div class="toolbar">
      <a-button type="primary" @click="showCreateModal">
        <PlusOutlined /> 创建知识库
      </a-button>
    </div>

    <!-- 知识库列表 -->
    <a-table
      :columns="columns"
      :data-source="knowledgeList"
      :loading="loading"
      @change="handleTableChange"
    >
      <!-- 操作列 -->
      <template #action="{ record }">
        <a-space>
          <a-button size="small" @click="viewDetail(record.kid)">
            查看
          </a-button>
          <a-button size="small" @click="editKnowledge(record)">
            编辑
          </a-button>
          <a-popconfirm
            title="确定删除该知识库吗？将同时删除所有附件和向量数据"
            @confirm="deleteKnowledge(record.kid)"
          >
            <a-button size="small" danger>删除</a-button>
          </a-popconfirm>
        </a-space>
      </template>
    </a-table>

    <!-- 知识库详情抽屉 -->
    <a-drawer
      v-model:visible="detailVisible"
      title="知识库详情"
      width="80%"
    >
      <!-- 基本信息 -->
      <a-descriptions :column="2" bordered>
        <a-descriptions-item label="知识库名称">
          {{ currentKnowledge.kname }}
        </a-descriptions-item>
        <a-descriptions-item label="嵌入模型">
          {{ currentKnowledge.embeddingModelName }}
        </a-descriptions-item>
        <a-descriptions-item label="块大小">
          {{ currentKnowledge.textBlockSize }}
        </a-descriptions-item>
        <a-descriptions-item label="重叠字符">
          {{ currentKnowledge.overlapChar }}
        </a-descriptions-item>
      </a-descriptions>

      <!-- 文件上传 -->
      <div class="upload-section">
        <h3>上传文档</h3>
        <a-upload
          :action="uploadUrl"
          :headers="uploadHeaders"
          :data="{ kid: currentKnowledge.kid }"
          :show-upload-list="false"
          @change="handleUploadChange"
        >
          <a-button>
            <UploadOutlined /> 选择文件
          </a-button>
        </a-upload>
        <p class="tips">支持: PDF, Word, Excel, TXT (最大50MB)</p>
      </div>

      <!-- 附件列表 -->
      <div class="attachment-section">
        <h3>已上传文档 ({{ attachmentList.length }})</h3>
        <a-table
          :columns="attachmentColumns"
          :data-source="attachmentList"
          :pagination="false"
        >
          <!-- 状态列 -->
          <template #vectorStatus="{ record }">
            <a-tag v-if="record.vectorStatus === 10" color="default">
              未开始
            </a-tag>
            <a-tag v-else-if="record.vectorStatus === 20" color="processing">
              处理中
            </a-tag>
            <a-tag v-else-if="record.vectorStatus === 30" color="success">
              已完成
            </a-tag>
            <a-tag v-else color="error">失败</a-tag>
          </template>

          <!-- 操作列 -->
          <template #action="{ record }">
            <a-space>
              <a-button size="small" @click="viewFragments(record.docId)">
                查看片段
              </a-button>
              <a-popconfirm
                title="确定删除该文档吗？将同时删除向量数据"
                @confirm="deleteAttachment(record.id)"
              >
                <a-button size="small" danger>删除</a-button>
              </a-popconfirm>
            </a-space>
          </template>
        </a-table>
      </div>

      <!-- 知识片段列表 -->
      <div class="fragment-section">
        <h3>知识片段 ({{ fragmentList.length }})</h3>
        <a-list
          :data-source="fragmentList"
          :pagination="{ pageSize: 10 }"
        >
          <template #renderItem="{ item }">
            <a-list-item>
              <a-list-item-meta>
                <template #title>
                  片段 #{{ item.idx }} (ID: {{ item.fid }})
                </template>
                <template #description>
                  <div class="fragment-content">
                    {{ item.content }}
                  </div>
                </template>
              </a-list-item-meta>
            </a-list-item>
          </template>
        </a-list>
      </div>
    </a-drawer>

    <!-- 创建/编辑知识库模态框 -->
    <a-modal
      v-model:visible="modalVisible"
      :title="isEdit ? '编辑知识库' : '创建知识库'"
      @ok="handleSubmit"
    >
      <a-form :model="formData" :label-col="{ span: 6 }">
        <a-form-item label="知识库名称" required>
          <a-input v-model:value="formData.kname" />
        </a-form-item>

        <a-form-item label="描述">
          <a-textarea v-model:value="formData.description" :rows="3" />
        </a-form-item>

        <a-form-item label="嵌入模型">
          <a-select v-model:value="formData.embeddingModelName">
            <a-select-option value="text-embedding-3-small">
              OpenAI - text-embedding-3-small (1536维)
            </a-select-option>
            <a-select-option value="bge-m3">
              Ollama - bge-m3 (1024维)
            </a-select-option>
            <a-select-option value="embedding-2">
              智谱AI - embedding-2 (1024维)
            </a-select-option>
          </a-select>
        </a-form-item>

        <a-form-item label="文本块大小">
          <a-input-number
            v-model:value="formData.textBlockSize"
            :min="100"
            :max="2000"
            :step="100"
          />
          <span class="tips">字符数，建议300-800</span>
        </a-form-item>

        <a-form-item label="重叠字符数">
          <a-input-number
            v-model:value="formData.overlapChar"
            :min="0"
            :max="200"
            :step="10"
          />
          <span class="tips">相邻块的重叠部分，建议10-20%</span>
        </a-form-item>

        <a-form-item label="检索条数">
          <a-input-number
            v-model:value="formData.retrieveLimit"
            :min="1"
            :max="20"
          />
          <span class="tips">默认返回的相似片段数</span>
        </a-form-item>

        <a-form-item label="是否公开">
          <a-switch v-model:checked="formData.share" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue';
import { message } from 'ant-design-vue';
import {
  knowledgeList,
  knowledgeSave,
  knowledgeDetail,
  knowledgeDelete,
  knowledgeFileDelete
} from '#/api/operator/knowledgeBase';
import { useAccessStore } from '#/store';

// 状态管理
const loading = ref(false);
const knowledgeList = ref([]);
const attachmentList = ref([]);
const fragmentList = ref([]);
const detailVisible = ref(false);
const modalVisible = ref(false);
const isEdit = ref(false);
const currentKnowledge = ref({});

// 表单数据
const formData = reactive({
  kid: '',
  kname: '',
  description: '',
  embeddingModelName: 'text-embedding-3-small',
  textBlockSize: 500,
  overlapChar: 50,
  retrieveLimit: 5,
  share: false
});

// 上传配置
const accessStore = useAccessStore();
const uploadUrl = computed(() => `${import.meta.env.VITE_API_URL}/knowledge/attach/upload`);
const uploadHeaders = computed(() => ({
  Authorization: `Bearer ${accessStore.accessToken}`,
  clientId: import.meta.env.VITE_CLIENT_ID
}));

// 表格列配置
const columns = [
  { title: 'ID', dataIndex: 'id', width: 80 },
  { title: '知识库名称', dataIndex: 'kname' },
  { title: '嵌入模型', dataIndex: 'embeddingModelName' },
  { title: '文档数', dataIndex: 'attachCount', width: 100 },
  { title: '片段数', dataIndex: 'fragmentCount', width: 100 },
  { title: '创建时间', dataIndex: 'createTime', width: 180 },
  { title: '操作', key: 'action', width: 250, slots: { customRender: 'action' } }
];

const attachmentColumns = [
  { title: '文档名称', dataIndex: 'docName' },
  { title: '类型', dataIndex: 'docType', width: 100 },
  { title: '向量化状态', key: 'vectorStatus', width: 120, slots: { customRender: 'vectorStatus' } },
  { title: '上传时间', dataIndex: 'createTime', width: 180 },
  { title: '操作', key: 'action', width: 180, slots: { customRender: 'action' } }
];

// 加载知识库列表
async function loadKnowledgeList() {
  loading.value = true;
  try {
    const { data } = await knowledgeList();
    knowledgeList.value = data;
  } catch (error) {
    message.error('加载失败');
  } finally {
    loading.value = false;
  }
}

// 查看详情
async function viewDetail(kid: string) {
  try {
    const { data } = await knowledgeDetail(kid);
    currentKnowledge.value = data;
    attachmentList.value = data.attachments || [];
    fragmentList.value = data.fragments || [];
    detailVisible.value = true;
  } catch (error) {
    message.error('加载详情失败');
  }
}

// 上传回调
function handleUploadChange(info: any) {
  if (info.file.status === 'done') {
    if (info.file.response.code === 200) {
      message.success('上传成功，正在处理...');
      // 刷新详情
      viewDetail(currentKnowledge.value.kid);
    } else {
      message.error(info.file.response.msg || '上传失败');
    }
  } else if (info.file.status === 'error') {
    message.error('上传失败');
  }
}

// 删除附件
async function deleteAttachment(id: number) {
  try {
    await knowledgeFileDelete(id);
    message.success('删除成功');
    viewDetail(currentKnowledge.value.kid);
  } catch (error) {
    message.error('删除失败');
  }
}

// 删除知识库
async function deleteKnowledge(kid: string) {
  try {
    await knowledgeDelete(kid);
    message.success('删除成功');
    loadKnowledgeList();
  } catch (error) {
    message.error('删除失败');
  }
}

// 创建/编辑
function showCreateModal() {
  isEdit.value = false;
  Object.assign(formData, {
    kid: '',
    kname: '',
    description: '',
    embeddingModelName: 'text-embedding-3-small',
    textBlockSize: 500,
    overlapChar: 50,
    retrieveLimit: 5,
    share: false
  });
  modalVisible.value = true;
}

function editKnowledge(record: any) {
  isEdit.value = true;
  Object.assign(formData, record);
  modalVisible.value = true;
}

async function handleSubmit() {
  try {
    await knowledgeSave({
      ...formData,
      share: formData.share ? 1 : 0
    });
    message.success(isEdit.value ? '更新成功' : '创建成功');
    modalVisible.value = false;
    loadKnowledgeList();
  } catch (error) {
    message.error('操作失败');
  }
}

// 初始化
onMounted(() => {
  loadKnowledgeList();
});
</script>

<style scoped lang="less">
.knowledge-base-container {
  padding: 20px;

  .toolbar {
    margin-bottom: 16px;
  }

  .upload-section,
  .attachment-section,
  .fragment-section {
    margin-top: 24px;

    h3 {
      margin-bottom: 16px;
      font-size: 16px;
      font-weight: 600;
    }

    .tips {
      margin-top: 8px;
      color: #999;
      font-size: 12px;
    }
  }

  .fragment-content {
    max-height: 100px;
    overflow: auto;
    padding: 8px;
    background: #f5f5f5;
    border-radius: 4px;
    font-size: 13px;
    line-height: 1.6;
  }
}
</style>
```

---

## 后端实现

### 核心服务实现

#### 知识库服务
**文件**: `backend/ruoyi-ai/ruoyi-modules/ruoyi-chat/src/main/java/org/ruoyi/chat/service/knowledge/KnowledgeInfoServiceImpl.java`

```java
@Service
@RequiredArgsConstructor
public class KnowledgeInfoServiceImpl implements IKnowledgeInfoService {

    private final KnowledgeInfoMapper knowledgeInfoMapper;
    private final KnowledgeFragmentMapper fragmentMapper;
    private final KnowledgeAttachMapper attachMapper;
    private final VectorStoreService vectorStoreService;
    private final ResourceLoaderFactory resourceLoaderFactory;

    /**
     * 创建知识库
     */
    @Transactional
    public void saveOne(KnowledgeInfoBo bo) {
        KnowledgeInfo entity = BeanUtil.toBean(bo, KnowledgeInfo.class);

        if (StringUtils.isBlank(bo.getKid())) {
            // 新建知识库
            String kid = RandomUtil.randomString(10);
            entity.setKid(kid);
            entity.setUid(LoginHelper.getUserId());
            knowledgeInfoMapper.insert(entity);

            // 在向量库中创建集合
            vectorStoreService.createSchema(
                String.valueOf(entity.getId()),
                bo.getEmbeddingModelName()
            );

            log.info("Created knowledge base: {}, collection: knowledge_{}",
                     entity.getKname(), kid);
        } else {
            // 更新知识库
            knowledgeInfoMapper.updateById(entity);
        }
    }

    /**
     * 上传文档并向量化
     */
    @Transactional
    public void storeContent(MultipartFile file, String kid) throws Exception {
        // 1. 验证知识库
        KnowledgeInfo knowledge = knowledgeInfoMapper.selectOne(
            new LambdaQueryWrapper<KnowledgeInfo>()
                .eq(KnowledgeInfo::getKid, kid)
        );
        if (knowledge == null) {
            throw new ServiceException("知识库不存在");
        }

        // 2. 检测文件类型
        String fileName = file.getOriginalFilename();
        String extension = fileName.substring(fileName.lastIndexOf(".") + 1);
        DocType docType = DocType.fromExtension(extension);

        // 3. 创建附件记录
        String docId = RandomUtil.randomString(10);
        KnowledgeAttach attach = new KnowledgeAttach();
        attach.setKid(kid);
        attach.setDocId(docId);
        attach.setDocName(fileName);
        attach.setDocType(docType.name());
        attach.setVectorStatus(20);  // 处理中
        attachMapper.insert(attach);

        try {
            // 4. 加载文档内容
            ResourceLoader loader = resourceLoaderFactory.getLoaderByFileType(docType);
            String content = loader.getContent(file.getInputStream());

            // 保存完整内容
            attach.setContent(content);
            attachMapper.updateById(attach);

            // 5. 文本分块
            List<String> chunks = loader.getChunkList(content, kid);
            log.info("Document split into {} chunks", chunks.size());

            // 6. 保存知识片段
            List<KnowledgeFragment> fragments = new ArrayList<>();
            List<String> fidList = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                String fid = RandomUtil.randomString(10);
                fidList.add(fid);

                KnowledgeFragment fragment = new KnowledgeFragment();
                fragment.setKid(kid);
                fragment.setDocId(docId);
                fragment.setFid(fid);
                fragment.setIdx(i);
                fragment.setContent(chunks.get(i));
                fragments.add(fragment);
            }
            fragmentMapper.insertBatch(fragments);

            // 7. 向量化并存储
            StoreEmbeddingBo embeddingBo = StoreEmbeddingBo.builder()
                .kid(kid)
                .docId(docId)
                .chunkList(chunks)
                .fidList(fidList)
                .embeddingModelName(knowledge.getEmbeddingModelName())
                .dimension(getModelDimension(knowledge.getEmbeddingModelName()))
                .build();

            vectorStoreService.storeEmbeddings(embeddingBo);

            // 8. 更新状态为完成
            attach.setVectorStatus(30);
            attachMapper.updateById(attach);

            log.info("Document processed successfully: {}", fileName);

        } catch (Exception e) {
            // 失败时更新状态
            attach.setVectorStatus(40);
            attachMapper.updateById(attach);

            log.error("Failed to process document: {}", fileName, e);
            throw new ServiceException("文档处理失败: " + e.getMessage());
        }
    }

    /**
     * 删除文档(同时删除向量)
     */
    @Transactional
    public void removeAttach(Long attachId) {
        KnowledgeAttach attach = attachMapper.selectById(attachId);
        if (attach == null) {
            throw new ServiceException("附件不存在");
        }

        try {
            // 删除向量数据
            vectorStoreService.removeByDocId(attach.getDocId(), attach.getKid());

            // 删除知识片段
            fragmentMapper.delete(
                new LambdaQueryWrapper<KnowledgeFragment>()
                    .eq(KnowledgeFragment::getDocId, attach.getDocId())
            );

            // 删除附件记录
            attachMapper.deleteById(attachId);

            log.info("Removed document: {}, docId: {}", attach.getDocName(), attach.getDocId());
        } catch (Exception e) {
            log.error("Failed to remove document", e);
            throw new ServiceException("删除失败");
        }
    }

    /**
     * 删除知识库(级联删除所有数据)
     */
    @Transactional
    public void removeKnowledge(String kid) {
        KnowledgeInfo knowledge = knowledgeInfoMapper.selectOne(
            new LambdaQueryWrapper<KnowledgeInfo>()
                .eq(KnowledgeInfo::getKid, kid)
        );
        if (knowledge == null) {
            throw new ServiceException("知识库不存在");
        }

        try {
            // 删除所有附件(会级联删除向量)
            List<KnowledgeAttach> attaches = attachMapper.selectList(
                new LambdaQueryWrapper<KnowledgeAttach>()
                    .eq(KnowledgeAttach::getKid, kid)
            );
            for (KnowledgeAttach attach : attaches) {
                removeAttach(attach.getId());
            }

            // 删除知识库记录
            knowledgeInfoMapper.deleteById(knowledge.getId());

            // 删除Milvus集合
            vectorStoreService.dropSchema(kid);

            log.info("Removed knowledge base: {}, kid: {}", knowledge.getKname(), kid);
        } catch (Exception e) {
            log.error("Failed to remove knowledge base", e);
            throw new ServiceException("删除知识库失败");
        }
    }

    /**
     * 查询知识库详情(包含附件和片段)
     */
    public KnowledgeInfoVo queryById(Long id) {
        KnowledgeInfo knowledge = knowledgeInfoMapper.selectById(id);
        if (knowledge == null) {
            return null;
        }

        KnowledgeInfoVo vo = BeanUtil.toBean(knowledge, KnowledgeInfoVo.class);

        // 附件列表
        List<KnowledgeAttach> attaches = attachMapper.selectList(
            new LambdaQueryWrapper<KnowledgeAttach>()
                .eq(KnowledgeAttach::getKid, knowledge.getKid())
        );
        vo.setAttachments(attaches);
        vo.setAttachCount(attaches.size());

        // 片段列表
        List<KnowledgeFragment> fragments = fragmentMapper.selectList(
            new LambdaQueryWrapper<KnowledgeFragment>()
                .eq(KnowledgeFragment::getKid, knowledge.getKid())
                .orderByAsc(KnowledgeFragment::getIdx)
        );
        vo.setFragments(fragments);
        vo.setFragmentCount(fragments.size());

        return vo;
    }

    /**
     * 检索知识库
     */
    public List<String> retrieve(String kid, String query, Integer topK) {
        KnowledgeInfo knowledge = knowledgeInfoMapper.selectOne(
            new LambdaQueryWrapper<KnowledgeInfo>()
                .eq(KnowledgeInfo::getKid, kid)
        );

        if (knowledge == null) {
            throw new ServiceException("知识库不存在");
        }

        QueryVectorBo queryBo = QueryVectorBo.builder()
            .kid(kid)
            .query(query)
            .maxResults(topK != null ? topK : knowledge.getRetrieveLimit())
            .embeddingModelName(knowledge.getEmbeddingModelName())
            .build();

        return vectorStoreService.getQueryVector(queryBo);
    }

    private Integer getModelDimension(String modelName) {
        // 从配置中获取模型维度
        ChatModelVo model = chatModelService.selectModelByName(modelName);
        return model.getDimension();
    }
}
```

---

## 向量数据库集成

### Milvus集成

#### 依赖配置
```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-milvus</artifactId>
    <version>0.27.1</version>
</dependency>
```

#### 配置类
```java
@Configuration
@ConfigurationProperties(prefix = "vector-store")
@Data
public class VectorStoreProperties {

    private String type = "milvus";  // 向量库类型

    private MilvusConfig milvus = new MilvusConfig();

    @Data
    public static class MilvusConfig {
        private String url = "http://localhost:19530";
        private String collectionname = "knowledge_";
    }
}
```

#### 策略实现
```java
@Component
@RequiredArgsConstructor
public class MilvusVectorStoreStrategy implements VectorStoreStrategy {

    private final VectorStoreProperties properties;
    private final EmbeddingModelFactory embeddingModelFactory;

    // 缓存EmbeddingStore实例
    private final Map<String, EmbeddingStore<TextSegment>> storeCache = new ConcurrentHashMap<>();

    /**
     * 获取或创建MilvusEmbeddingStore
     */
    private EmbeddingStore<TextSegment> getMilvusStore(String collectionName,
                                                        int dimension,
                                                        boolean create) {
        return storeCache.computeIfAbsent(collectionName, name -> {
            MilvusEmbeddingStore.Builder builder = MilvusEmbeddingStore.builder()
                .uri(properties.getMilvus().getUrl())
                .collectionName(collectionName)
                .dimension(dimension)
                .indexType(IndexType.IVF_FLAT)
                .metricType(MetricType.L2)
                .autoFlushOnInsert(true);

            if (!create) {
                // 仅连接，不创建
                builder.createCollection(false);
            }

            return builder.build();
        });
    }

    @Override
    public void createSchema(String kid, String embeddingModelName) {
        String collectionName = properties.getMilvus().getCollectionname() + kid;
        int dimension = getModelDimension(embeddingModelName);

        getMilvusStore(collectionName, dimension, true);

        log.info("Created Milvus collection: {}, dimension: {}", collectionName, dimension);
    }

    @Override
    public void storeEmbeddings(StoreEmbeddingBo bo) {
        String collectionName = properties.getMilvus().getCollectionname() + bo.getKid();

        EmbeddingModel embeddingModel = embeddingModelFactory.createModel(
            bo.getEmbeddingModelName(),
            bo.getDimension()
        );

        EmbeddingStore<TextSegment> store = getMilvusStore(
            collectionName,
            bo.getDimension(),
            false
        );

        for (int i = 0; i < bo.getChunkList().size(); i++) {
            String chunk = bo.getChunkList().get(i);
            String fid = bo.getFidList().get(i);

            // 生成向量
            Embedding embedding = embeddingModel.embed(chunk).content();

            // 创建文本段(带元数据)
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("kid", bo.getKid());
            metadata.put("docId", bo.getDocId());
            metadata.put("fid", fid);
            metadata.put("idx", i);

            TextSegment segment = TextSegment.from(chunk, Metadata.from(metadata));

            // 存储
            store.add(embedding, segment);
        }

        log.info("Stored {} embeddings to collection: {}",
                 bo.getChunkList().size(), collectionName);
    }

    @Override
    public List<String> getQueryVector(QueryVectorBo bo) {
        String collectionName = properties.getMilvus().getCollectionname() + bo.getKid();

        EmbeddingModel embeddingModel = embeddingModelFactory.createModel(
            bo.getEmbeddingModelName(),
            null
        );

        EmbeddingStore<TextSegment> store = storeCache.get(collectionName);
        if (store == null) {
            throw new ServiceException("Collection not found: " + collectionName);
        }

        // 向量化查询
        Embedding queryEmbedding = embeddingModel.embed(bo.getQuery()).content();

        // 构建搜索请求
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
            .queryEmbedding(queryEmbedding)
            .maxResults(bo.getMaxResults())
            .minScore(0.7)  // 最小相似度
            .build();

        // 执行搜索
        EmbeddingSearchResult<TextSegment> result = store.search(request);

        // 提取结果
        List<String> results = new ArrayList<>();
        for (EmbeddingMatch<TextSegment> match : result.matches()) {
            TextSegment segment = match.embedded();
            if (segment != null) {
                results.add(segment.text());
                log.debug("Match score: {}, fid: {}",
                         match.score(),
                         segment.metadata().getString("fid"));
            }
        }

        log.info("Retrieved {} results for query: {}", results.size(), bo.getQuery());

        return results;
    }

    @Override
    public void removeByDocId(String docId, String kid) {
        String collectionName = properties.getMilvus().getCollectionname() + kid;

        EmbeddingStore<TextSegment> store = storeCache.get(collectionName);
        if (store == null) {
            return;
        }

        // 构建过滤条件
        Filter filter = MetadataFilterBuilder
            .metadataKey("docId")
            .isEqualTo(docId);

        // 删除匹配的向量
        store.removeAll(filter);

        log.info("Removed embeddings for docId: {} from collection: {}",
                 docId, collectionName);
    }

    /**
     * 删除集合
     */
    public void dropSchema(String kid) {
        String collectionName = properties.getMilvus().getCollectionname() + kid;
        storeCache.remove(collectionName);

        // 调用Milvus API删除集合
        // 注意: langchain4j-milvus可能不直接支持dropCollection
        // 需要使用Milvus SDK直接操作

        log.info("Dropped collection: {}", collectionName);
    }
}
```

---

## 配置指南

### 1. 数据库配置

**application-dev.yml**:
```yaml
spring:
  datasource:
    dynamic:
      primary: master
      datasource:
        master:
          url: jdbc:mysql://127.0.0.1:3306/ruoyi-ai?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull&useSSL=true&serverTimezone=GMT%2B8
          username: root
          password: your_password
          driver-class-name: com.mysql.cj.jdbc.Driver
```

### 2. 向量数据库配置

#### Milvus配置
```yaml
vector-store:
  type: milvus
  milvus:
    url: http://localhost:19530
    collectionname: knowledge_
```

**Milvus Docker部署**:
```bash
# 下载docker-compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动Milvus
docker-compose up -d

# 查看状态
docker-compose ps
```

#### Weaviate配置
```yaml
vector-store:
  type: weaviate
  weaviate:
    protocol: http
    host: localhost:8080
    classname: KnowledgeClass
```

**Weaviate Docker部署**:
```bash
docker run -d \
  -p 8080:8080 \
  --name weaviate \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

### 3. 嵌入模型配置

#### 数据库表配置
在 `chat_model` 表中添加嵌入模型:

```sql
INSERT INTO chat_model (
  id, model_name, model_type, provider_name,
  api_host, api_key, dimension, priority
) VALUES
  (1, 'text-embedding-3-small', 'embedding', 'openai',
   'https://api.openai.com', 'sk-your-key', 1536, 1),
  (2, 'bge-m3', 'embedding', 'ollama',
   'http://localhost:11434', '', 1024, 2),
  (3, 'embedding-2', 'embedding', 'zhipuai',
   'https://open.bigmodel.cn', 'your-key', 1024, 3);
```

#### Ollama本地部署
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull bge-m3

# 查看已安装模型
ollama list

# 测试嵌入
curl http://localhost:11434/api/embeddings -d '{
  "model": "bge-m3",
  "prompt": "测试文本"
}'
```

### 4. 文件上传配置

```yaml
spring:
  servlet:
    multipart:
      enabled: true
      max-file-size: 50MB        # 单个文件最大大小
      max-request-size: 200MB    # 总请求最大大小

server:
  undertow:
    max-http-post-size: -1       # 无限制
```

### 5. Redis配置(缓存嵌入模型)

```yaml
spring:
  data:
    redis:
      host: 127.0.0.1
      port: 6379
      database: 0
      password:
      timeout: 10s
      lettuce:
        pool:
          max-active: 200
          max-idle: 10
          min-idle: 0
          max-wait: -1ms
```

---

## API接口说明

### 知识库管理API

#### 1. 获取知识库列表
```http
GET /knowledge/list

Response:
{
  "code": 200,
  "data": [
    {
      "id": 1,
      "kid": "kid_abc123",
      "kname": "技术文档库",
      "embeddingModelName": "text-embedding-3-small",
      "attachCount": 5,
      "fragmentCount": 120,
      "createTime": "2024-01-01 10:00:00"
    }
  ]
}
```

#### 2. 创建知识库
```http
POST /knowledge/save
Content-Type: application/json

{
  "kname": "新知识库",
  "description": "用于存储RAG相关文档",
  "embeddingModelName": "text-embedding-3-small",
  "textBlockSize": 500,
  "overlapChar": 50,
  "retrieveLimit": 5,
  "share": 0
}

Response:
{
  "code": 200,
  "msg": "操作成功"
}
```

#### 3. 获取知识库详情
```http
GET /knowledge/detail/{id}

Response:
{
  "code": 200,
  "data": {
    "id": 1,
    "kid": "kid_abc123",
    "kname": "技术文档库",
    "embeddingModelName": "text-embedding-3-small",
    "textBlockSize": 500,
    "overlapChar": 50,
    "attachments": [
      {
        "id": 1001,
        "docId": "doc_xyz789",
        "docName": "RAG技术文档.pdf",
        "docType": "pdf",
        "vectorStatus": 30,
        "createTime": "2024-01-01 10:00:00"
      }
    ],
    "fragments": [
      {
        "id": 2001,
        "fid": "frag_001",
        "idx": 0,
        "content": "这是第一个文本片段..."
      }
    ]
  }
}
```

#### 4. 上传文档
```http
POST /knowledge/attach/upload
Content-Type: multipart/form-data

Parameters:
  file: (binary)
  kid: "kid_abc123"

Response:
{
  "code": 200,
  "msg": "上传成功",
  "data": {
    "docId": "doc_xyz789",
    "docName": "document.pdf",
    "chunkCount": 25
  }
}
```

#### 5. 删除文档
```http
POST /knowledge/attach/remove/{id}

Response:
{
  "code": 200,
  "msg": "删除成功"
}
```

#### 6. 删除知识库
```http
POST /knowledge/remove/{kid}

Response:
{
  "code": 200,
  "msg": "删除成功"
}
```

#### 7. 获取知识片段列表
```http
GET /knowledge/fragment/list?kid=kid_abc123&docId=doc_xyz789

Response:
{
  "code": 200,
  "data": [
    {
      "id": 2001,
      "fid": "frag_001",
      "idx": 0,
      "content": "片段内容...",
      "createTime": "2024-01-01 10:05:00"
    }
  ]
}
```

### 知识检索API

#### 8. 检索知识库
```http
POST /knowledge/retrieve
Content-Type: application/json

{
  "kid": "kid_abc123",
  "query": "RAG的核心组件有哪些?",
  "topK": 5
}

Response:
{
  "code": 200,
  "data": {
    "results": [
      "RAG系统主要包含三个核心组件: 检索器、生成器和知识库...",
      "检索增强生成由文档处理、向量存储和语言模型三部分组成...",
      "RAG的关键技术包括嵌入模型、向量数据库和相似度搜索..."
    ],
    "count": 3
  }
}
```

---

## 部署与使用

### 快速开始

#### 1. 环境准备
```bash
# Java 17+
java -version

# MySQL 8.0+
mysql --version

# Redis
redis-cli ping

# Milvus (Docker)
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.3.0
```

#### 2. 数据库初始化
```bash
# 创建数据库
mysql -u root -p -e "CREATE DATABASE ruoyi_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 导入表结构
mysql -u root -p ruoyi_ai < backend/ruoyi-ai/script/sql/ruoyi-ai.sql
```

#### 3. 配置文件
编辑 `ruoyi-admin/src/main/resources/application-dev.yml`:
```yaml
# 数据库
spring.datasource.dynamic.datasource.master.url: jdbc:mysql://localhost:3306/ruoyi_ai
spring.datasource.dynamic.datasource.master.username: root
spring.datasource.dynamic.datasource.master.password: your_password

# Redis
spring.data.redis.host: localhost
spring.data.redis.port: 6379

# 向量数据库
vector-store.type: milvus
vector-store.milvus.url: http://localhost:19530
```

#### 4. 启动后端
```bash
cd backend/ruoyi-ai/ruoyi-admin
mvn spring-boot:run
```

#### 5. 启动前端
```bash
cd frontend/ruoyi-admin
pnpm install
pnpm dev
```

#### 6. 访问系统
```
前端地址: http://localhost:5173
后端地址: http://localhost:6039
```

### 使用流程

#### 创建知识库
1. 登录系统
2. 进入「知识库管理」
3. 点击「创建知识库」
4. 填写配置:
   - 知识库名称: "技术文档库"
   - 嵌入模型: "text-embedding-3-small"
   - 块大小: 500
   - 重叠字符: 50
   - 检索条数: 5
5. 提交创建

#### 上传文档
1. 打开知识库详情
2. 点击「上传文件」
3. 选择文档(支持PDF/Word/Excel/TXT)
4. 等待处理完成(自动分块、向量化)
5. 查看知识片段

#### 使用检索
**方式1: 工作流节点**
```yaml
节点类型: knowledgeRetrieval
配置:
  knowledgeId: "kid_abc123"
  topK: 5
  retrievalMode: "vector"
  embeddingModel: "text-embedding-3-small"
```

**方式2: API调用**
```javascript
const response = await fetch('/knowledge/retrieve', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    kid: 'kid_abc123',
    query: '如何实现RAG?',
    topK: 5
  })
});

const { data } = await response.json();
console.log(data.results);
```

**方式3: 工作流集成**
```
用户输入 → 知识检索节点 → LLM生成节点 → 返回答案

工作流配置:
1. 知识检索节点:
   - 输入: {{user_input}}
   - 检索Top 5相关片段

2. LLM生成节点:
   - 输入: "根据以下知识回答问题:\n{{retrieval_result}}\n\n问题: {{user_input}}"
   - 生成最终答案
```

---

## 性能优化建议

### 1. 分块策略
```
小块(200-300字符):
  优点: 检索精确度高
  缺点: 上下文不足，需要更多token
  适用: 问答系统、精确查找

中块(500-800字符):
  优点: 平衡精确度和上下文
  缺点: 需要适当重叠
  适用: 通用场景(推荐)

大块(1000-1500字符):
  优点: 上下文丰富
  缺点: 检索精度降低
  适用: 长文本理解、摘要生成
```

### 2. 向量维度选择
```
低维(384-512):
  优点: 存储小、查询快
  缺点: 表达能力有限

中维(768-1024):
  优点: 平衡性能和效果
  适用: 大多数场景

高维(1536-3072):
  优点: 最佳效果
  缺点: 存储和计算成本高
  适用: 高质量要求场景
```

### 3. 索引优化
```
Milvus索引选择:
- FLAT: 精确搜索，<10K向量
- IVF_FLAT: 中等规模，10K-1M向量
- IVF_SQ8: 大规模，>1M向量，压缩存储
- HNSW: 极速查询，内存充足
```

### 4. 缓存策略
```java
// 嵌入模型缓存
@Cacheable(value = "embedding", key = "#text")
public Embedding embed(String text) {
    return embeddingModel.embed(text).content();
}

// 检索结果缓存
@Cacheable(value = "retrieval", key = "#kid + ':' + #query")
public List<String> retrieve(String kid, String query) {
    return vectorStoreService.getQueryVector(...);
}
```

---

## 故障排查

### 常见问题

#### 1. 向量化失败
```
错误: Failed to embed text
原因: 嵌入模型API密钥无效或网络问题
解决:
  - 检查chat_model表中的apiKey
  - 测试API连接: curl https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_KEY"
  - 使用Ollama本地模型作为备选
```

#### 2. Milvus连接失败
```
错误: Connection refused: localhost:19530
原因: Milvus服务未启动
解决:
  docker-compose ps  # 检查状态
  docker-compose up -d  # 启动服务
  docker logs milvus-standalone  # 查看日志
```

#### 3. 文档分块失败
```
错误: Split text failed
原因: 文档格式不支持或损坏
解决:
  - 检查文件是否完整
  - 尝试转换为纯文本格式
  - 调整textBlockSize参数
```

#### 4. 检索结果为空
```
原因:
  1. 相似度阈值过高
  2. 嵌入模型不匹配
  3. 集合不存在

解决:
  - 降低similarityThreshold
  - 确认embeddingModelName一致
  - 检查Milvus集合: show collections;
```

### 日志查看
```bash
# 后端日志
tail -f backend/ruoyi-ai/logs/sys-info.log

# Milvus日志
docker logs -f milvus-standalone

# Redis监控
redis-cli monitor
```

---

## 总结

本RAG系统实现了完整的知识库管理和检索功能:

**核心特性**:
- ✅ 多文档类型支持(PDF/Word/Excel/TXT)
- ✅ 灵活的分块策略(自定义分隔符、重叠)
- ✅ 多嵌入模型支持(OpenAI/Ollama/智谱AI等)
- ✅ 向量数据库集成(Milvus/Weaviate)
- ✅ 工作流无缝集成
- ✅ 权限控制和角色管理

**技术亮点**:
- 策略模式实现向量库切换
- 工厂模式管理嵌入模型
- 完整的异步处理流程
- 元数据丰富的向量存储
- 高效的相似度搜索

**适用场景**:
- 企业知识库问答
- 文档智能检索
- 客服机器人
- 技术文档助手

---

## 文件位置索引

| 功能 | 后端路径 | 前端路径 |
|------|---------|---------|
| 知识库管理 | `ruoyi-modules/ruoyi-chat/src/main/java/org/ruoyi/chat/service/knowledge/` | `src/views/operator/knowledgeBase/` |
| 向量存储 | `ruoyi-modules-api/ruoyi-knowledge-api/src/main/java/org/ruoyi/service/` | - |
| 嵌入模型 | `ruoyi-modules-api/ruoyi-knowledge-api/src/main/java/org/ruoyi/embedding/` | - |
| 文档处理 | `ruoyi-modules-api/ruoyi-knowledge-api/src/main/java/org/ruoyi/chain/` | - |
| 工作流节点 | `ruoyi-modules-api/ruoyi-workflow-api/src/main/java/org/ruoyi/workflow/workflow/node/knowledgeRetrieval/` | - |
| 数据模型 | `ruoyi-modules-api/ruoyi-knowledge-api/src/main/java/org/ruoyi/domain/` | - |
| API接口 | `ruoyi-modules/ruoyi-chat/src/main/java/org/ruoyi/chat/controller/knowledge/` | `src/api/operator/knowledgeBase/` |
| 数据库脚本 | `script/sql/ruoyi-ai.sql` | - |

---

**文档版本**: 1.0
**最后更新**: 2024-01-29
**作者**: AI助手
