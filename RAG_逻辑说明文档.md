# RuoYi-AI RAG 逻辑说明文档

## 一、RAG 整体架构

### 1.1 什么是 RAG？

RAG (Retrieval-Augmented Generation) 是一种将检索与生成相结合的技术：
- **检索（Retrieval）**：从知识库中找到与用户问题最相关的文档片段
- **增强（Augmented）**：将检索到的内容作为上下文注入到提示词中
- **生成（Generation）**：让 LLM 基于这些上下文生成回答

### 1.2 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端 (Vue3)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  选择模型   │  │ 选择知识库  │  │  输入问题   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /chat/send
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     后端 (Spring Boot)                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   ChatController                             ││
│  │                        ↓                                     ││
│  │                   SseServiceImpl                             ││
│  │           ┌───────────┴───────────┐                         ││
│  │           ↓                       ↓                         ││
│  │   processKnowledgeBase()    autoSelectModel()               ││
│  │           ↓                       ↓                         ││
│  │   VectorStoreService        ChatServiceFactory              ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │  MySQL   │   │  Milvus  │   │   LLM    │
       │ 知识库配置│   │ 向量存储  │   │  API调用  │
       └──────────┘   └──────────┘   └──────────┘
```

---

## 二、前端逻辑

### 2.1 请求参数结构

```typescript
interface ChatRequest {
  messages: Message[];        // 消息列表
  model: string;              // 对话模型名称（如 gpt-4, deepseek-chat）
  kid?: string;               // 知识库ID（关键！有值则启用RAG）
  sessionId?: string;         // 会话ID
  autoSelectModel?: boolean;  // 是否自动选择模型
  stream?: boolean;           // 是否流式输出
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}
```

### 2.2 前端发起请求示例

```javascript
// 用户选择知识库后发起对话
const response = await fetch('/chat/send', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: '什么是RAG？' }],
    model: 'gpt-4',           // 用户选择的对话模型
    kid: '1234567890',        // 用户选择的知识库ID
    stream: true
  })
});
```

### 2.3 关键点说明

| 参数 | 说明 |
|------|------|
| `model` | **对话模型**，用于生成最终回答（如 GPT-4、DeepSeek） |
| `kid` | **知识库ID**，系统会根据此ID找到对应的Embedding模型配置 |

**重要**：`model` 和 `kid` 对应的是两个不同的模型：
- `model` → 对话/生成模型
- `kid` 关联的知识库 → 有自己独立的 Embedding 模型配置

---

## 三、后端逻辑详解

### 3.1 核心流程（SseServiceImpl）

```java
@PostMapping("/send")
public SseEmitter sseChat(@RequestBody ChatRequest chatRequest) {
    return sseService.sseChat(chatRequest, request);
}
```

#### 步骤1：处理知识库逻辑

```java
private String processKnowledgeBase(ChatRequest chatRequest, List<Message> messages) {
    // 1. 检查是否指定知识库
    if (StringUtils.isEmpty(chatRequest.getKid())) {
        return getDefaultPrompt();  // 无知识库，返回默认提示词
    }

    // 2. 查询知识库配置（包含Embedding模型信息）
    KnowledgeInfoVo knowledgeInfo = knowledgeInfoService.queryById(chatRequest.getKid());

    // 3. 获取知识库绑定的Embedding模型配置
    ChatModelVo embeddingModel = chatModelService.selectModelByName(
        knowledgeInfo.getEmbeddingModelName()  // ⭐ 使用知识库配置的Embedding模型
    );

    // 4. 构建向量查询参数
    QueryVectorBo queryVectorBo = new QueryVectorBo();
    queryVectorBo.setQuery(userQuestion);                          // 用户问题
    queryVectorBo.setKid(chatRequest.getKid());                    // 知识库ID
    queryVectorBo.setEmbeddingModelName(embeddingModel.getName()); // Embedding模型
    queryVectorBo.setApiKey(embeddingModel.getApiKey());           // API密钥
    queryVectorBo.setBaseUrl(embeddingModel.getApiHost());         // API地址

    // 5. 执行向量检索
    List<String> relevantDocs = vectorStoreService.getQueryVector(queryVectorBo);

    // 6. 将检索结果添加到上下文
    addKnowledgeMessages(messages, relevantDocs);

    return knowledgeInfo.getSystemPrompt();
}
```

#### 步骤2：向量检索实现

```java
// MilvusVectorStoreStrategy.java
public List<String> getQueryVector(QueryVectorBo queryVectorBo) {
    // 1. 创建Embedding模型（使用知识库配置的模型）
    EmbeddingModel embeddingModel = embeddingModelFactory.createModel(
        queryVectorBo.getEmbeddingModelName(),
        dimension
    );

    // 2. 将用户问题向量化
    Embedding queryEmbedding = embeddingModel.embed(queryVectorBo.getQuery()).content();

    // 3. 在向量库中搜索相似内容
    EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
        .queryEmbedding(queryEmbedding)
        .maxResults(queryVectorBo.getMaxResults())
        .build();

    // 4. 执行相似度搜索
    List<EmbeddingMatch<TextSegment>> matches = embeddingStore.search(request).matches();

    // 5. 返回最相关的文本片段
    return matches.stream()
        .map(m -> m.embedded().text())
        .collect(Collectors.toList());
}
```

#### 步骤3：构建最终消息并调用对话模型

```java
// 最终发送给对话模型的消息结构
List<Message> finalMessages = [
    {role: "system", content: "知识库系统提示词 + 检索到的相关文档"},
    {role: "user", content: "用户的原始问题"}
];

// 调用对话模型（用户选择的 model，如 gpt-4）
IChatService chatService = chatServiceFactory.getChatService(chatRequest.getModel());
chatService.chat(finalMessages, sseEmitter);
```

---

## 四、数据库设计

### 4.1 知识库表 (knowledge_info)

```sql
CREATE TABLE knowledge_info (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    kid VARCHAR(50) UNIQUE COMMENT '知识库唯一标识',
    uid BIGINT COMMENT '用户ID',
    kname VARCHAR(200) COMMENT '知识库名称',
    description TEXT COMMENT '描述',

    -- ⭐ Embedding模型配置（存储文档时使用）
    embedding_model_id BIGINT COMMENT 'Embedding模型ID',
    embedding_model_name VARCHAR(100) COMMENT 'Embedding模型名称',

    -- 向量库配置
    vector_model_name VARCHAR(100) COMMENT '向量库类型(milvus/weaviate)',

    -- 检索配置
    retrieve_limit INT DEFAULT 5 COMMENT '检索返回数量',
    text_block_size INT DEFAULT 512 COMMENT '文本块大小',
    overlap_char INT DEFAULT 100 COMMENT '重叠字符数',

    system_prompt TEXT COMMENT '自定义系统提示词',
    create_time DATETIME,
    update_time DATETIME
);
```

### 4.2 模型配置表 (chat_model)

```sql
CREATE TABLE chat_model (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(100) COMMENT '模型名称',
    provider_name VARCHAR(50) COMMENT '提供商(openai/ollama/zhipu等)',
    category VARCHAR(20) COMMENT '类型(chat/embedding/image)',
    api_key VARCHAR(500) COMMENT 'API密钥',
    api_host VARCHAR(200) COMMENT 'API地址',
    dimension INT COMMENT '向量维度(Embedding模型专用)',
    priority INT COMMENT '优先级',
    status TINYINT COMMENT '状态'
);
```

### 4.3 数据关系图

```
┌─────────────────────┐
│    knowledge_info   │
├─────────────────────┤
│ embedding_model_name│──────┐
│ vector_model_name   │      │
└─────────────────────┘      │
                             │
                             ▼
┌─────────────────────┐    ┌─────────────────────┐
│   chat_model        │    │   chat_model        │
│   (对话模型)         │    │   (Embedding模型)   │
├─────────────────────┤    ├─────────────────────┤
│ category = 'chat'   │    │ category='embedding'│
│ model_name='gpt-4'  │    │ model_name='bge-m3' │
└─────────────────────┘    └─────────────────────┘
       ↑                          ↑
       │                          │
  用户选择的模型             知识库绑定的模型
  (生成回答)                (向量化查询)
```

---

## 五、核心问题解答

### 5.1 用户选择的模型，必须和存储文档时的 Embedding 模型一致吗？

**答案：不需要一致，它们是两个完全不同的模型，各司其职。**

| 模型类型 | 用途 | 配置位置 | 示例 |
|---------|------|---------|------|
| **对话模型** | 生成最终回答 | 用户在前端选择 | GPT-4、DeepSeek、Qwen |
| **Embedding模型** | 向量化（存储和检索） | 知识库配置绑定 | text-embedding-3-small、bge-m3 |

**但是有一个关键约束**：
> **检索时使用的 Embedding 模型必须和存储文档时使用的 Embedding 模型一致！**

原因：
- 存储时：文档被 Embedding 模型 A 转换为向量 V1
- 检索时：用户问题被 Embedding 模型 B 转换为向量 V2
- 如果 A ≠ B，则 V1 和 V2 的向量空间不一致，相似度计算无意义

**代码实现**：系统自动保证这一点
```java
// 从知识库配置中获取 Embedding 模型（与存储时一致）
ChatModelVo embeddingModel = chatModelService.selectModelByName(
    knowledgeInfo.getEmbeddingModelName()  // 知识库创建时配置的模型
);
```

### 5.2 必须使用 Ollama 加载的模型吗？

**答案：不是必须的，系统支持多种 Embedding 模型提供商。**

支持的 Embedding 模型提供商：

| 提供商 | 示例模型 | 特点 |
|--------|---------|------|
| **OpenAI** | text-embedding-3-small/large | 效果好，需付费 |
| **Ollama** | nomic-embed-text, bge-m3 | 本地部署，免费 |
| **智谱AI** | embedding-2 | 中文效果好 |
| **硅基流动** | - | 国产替代 |
| **阿里百炼** | text-embedding-v2 | 中文优化 |

**配置示例**：
```yaml
# 使用 OpenAI Embedding（云端）
chat_model:
  model_name: text-embedding-3-small
  provider_name: openai
  api_host: https://api.openai.com/v1
  api_key: sk-xxx
  category: embedding
  dimension: 1536

# 使用 Ollama Embedding（本地）
chat_model:
  model_name: nomic-embed-text
  provider_name: ollama
  api_host: http://localhost:11434
  category: embedding
  dimension: 768
```

### 5.3 使用别的模型搜索知识库，为什么结果像是模型自己输出的？

**答案：这正是 RAG 的核心原理 —— 将检索结果作为上下文注入提示词。**

#### 工作流程图解

```
用户问题: "公司的年假政策是什么？"
                │
                ▼
┌─────────────────────────────────────┐
│  Step 1: 向量检索                    │
│  使用 Embedding 模型将问题向量化     │
│  在向量库中搜索相似文档              │
│                                     │
│  检索结果:                           │
│  1. "员工入职满一年可享受5天年假..." │
│  2. "年假需提前3天申请..."           │
│  3. "未休年假可折算工资..."          │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 2: 构建增强提示词              │
│                                     │
│  System: 你是一个企业助手，请根据    │
│  以下知识库内容回答用户问题：        │
│                                     │
│  【知识库内容】                      │
│  1. 员工入职满一年可享受5天年假...   │
│  2. 年假需提前3天申请...             │
│  3. 未休年假可折算工资...            │
│                                     │
│  User: 公司的年假政策是什么？        │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 3: 调用对话模型                │
│  GPT-4 / DeepSeek 等                │
│                                     │
│  模型基于注入的知识库内容生成回答    │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  最终输出:                           │
│  "根据公司规定，年假政策如下：       │
│   1. 入职满一年的员工可享受5天年假   │
│   2. 申请年假需提前3天提交           │
│   3. 未休完的年假可以折算成工资..."  │
└─────────────────────────────────────┘
```

#### 为什么看起来像模型"自己知道"的？

1. **上下文注入**：检索到的文档被放入 System Message，模型会基于这些内容回答
2. **自然语言生成**：LLM 会重新组织、总结知识库内容，使回答更流畅
3. **无痕融合**：用户看不到原始的检索结果，只看到最终的生成回答

#### 核心代码实现

```java
// SseServiceImpl.java
private void addKnowledgeMessages(List<Message> messages, List<String> nearestList) {
    if (CollectionUtils.isEmpty(nearestList)) {
        return;
    }

    // 将检索结果拼接成上下文
    StringBuilder knowledgeContext = new StringBuilder();
    knowledgeContext.append("请根据以下知识库内容回答用户问题：\n\n");

    for (int i = 0; i < nearestList.size(); i++) {
        knowledgeContext.append(String.format("【文档%d】\n%s\n\n", i + 1, nearestList.get(i)));
    }

    // 添加到系统消息
    Message systemMessage = Message.builder()
        .role(Message.Role.SYSTEM)
        .content(knowledgeContext.toString())
        .build();

    messages.add(0, systemMessage);
}
```

---

## 六、完整请求-响应流程

```
┌────────────────────────────────────────────────────────────────────────┐
│                           完整 RAG 流程                                 │
└────────────────────────────────────────────────────────────────────────┘

1. 前端请求
   ┌─────────────────────────────────────┐
   │ POST /chat/send                     │
   │ {                                   │
   │   model: "gpt-4",        ← 对话模型 │
   │   kid: "kb123",          ← 知识库ID │
   │   messages: [...]        ← 用户问题 │
   │ }                                   │
   └─────────────────────────────────────┘
                    │
                    ▼
2. 查询知识库配置 (MySQL)
   ┌─────────────────────────────────────┐
   │ SELECT * FROM knowledge_info        │
   │ WHERE kid = 'kb123'                 │
   │                                     │
   │ 结果:                               │
   │   embedding_model_name: 'bge-m3'    │
   │   vector_model_name: 'milvus'       │
   │   retrieve_limit: 5                 │
   └─────────────────────────────────────┘
                    │
                    ▼
3. 查询 Embedding 模型配置 (MySQL)
   ┌─────────────────────────────────────┐
   │ SELECT * FROM chat_model            │
   │ WHERE model_name = 'bge-m3'         │
   │                                     │
   │ 结果:                               │
   │   api_host: 'http://localhost:11434'│
   │   provider_name: 'ollama'           │
   │   dimension: 1024                   │
   └─────────────────────────────────────┘
                    │
                    ▼
4. 向量化用户问题 (Embedding API)
   ┌─────────────────────────────────────┐
   │ POST http://localhost:11434/embed   │
   │ { model: "bge-m3", input: "问题" }  │
   │                                     │
   │ 结果: [0.12, -0.34, 0.56, ...]     │
   │       (1024维向量)                  │
   └─────────────────────────────────────┘
                    │
                    ▼
5. 向量相似度搜索 (Milvus)
   ┌─────────────────────────────────────┐
   │ Collection: LocalKnowledge_kb123    │
   │ 搜索: 与查询向量最相似的5条记录     │
   │ 算法: L2距离 / 余弦相似度           │
   │                                     │
   │ 结果:                               │
   │   ["文档片段1", "文档片段2", ...]   │
   └─────────────────────────────────────┘
                    │
                    ▼
6. 构建增强消息
   ┌─────────────────────────────────────┐
   │ messages = [                        │
   │   {                                 │
   │     role: "system",                 │
   │     content: "知识库内容:\n" +      │
   │              "文档片段1\n" +        │
   │              "文档片段2\n..."       │
   │   },                                │
   │   {                                 │
   │     role: "user",                   │
   │     content: "用户原始问题"         │
   │   }                                 │
   │ ]                                   │
   └─────────────────────────────────────┘
                    │
                    ▼
7. 调用对话模型 (OpenAI API)
   ┌─────────────────────────────────────┐
   │ POST https://api.openai.com/v1/chat │
   │ {                                   │
   │   model: "gpt-4",                   │
   │   messages: [...],  ← 包含知识库内容│
   │   stream: true                      │
   │ }                                   │
   └─────────────────────────────────────┘
                    │
                    ▼
8. SSE 流式返回前端
   ┌─────────────────────────────────────┐
   │ data: {"content": "根据..."}        │
   │ data: {"content": "公司规定..."}    │
   │ data: {"content": "年假政策..."}    │
   │ data: [DONE]                        │
   └─────────────────────────────────────┘
```

---

## 七、配置建议

### 7.1 Embedding 模型选择

| 场景 | 推荐模型 | 提供商 | 维度 |
|------|---------|--------|------|
| 中文为主 | bge-m3, bge-large-zh | Ollama/智谱 | 1024 |
| 英文为主 | text-embedding-3-small | OpenAI | 1536 |
| 多语言 | bge-m3 | Ollama | 1024 |
| 成本敏感 | nomic-embed-text | Ollama | 768 |

### 7.2 知识库参数配置

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `text_block_size` | 512-1024 | 文本块大小，越小检索越精确 |
| `overlap_char` | 100-200 | 块之间重叠字符，防止信息丢失 |
| `retrieve_limit` | 5-10 | 检索返回数量，太多会超出上下文 |

### 7.3 模型搭配建议

```
┌─────────────────────────────────────────────────────────────┐
│  推荐组合                                                    │
├─────────────────────────────────────────────────────────────┤
│  Embedding: bge-m3 (Ollama本地)                             │
│  对话模型: GPT-4 / DeepSeek-V3                              │
│  向量库: Milvus                                             │
│                                                             │
│  优点:                                                      │
│  - Embedding 本地运行，无API费用                            │
│  - 对话模型使用云端高质量模型                               │
│  - Milvus 性能优秀，支持大规模数据                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、总结

| 问题 | 答案 |
|------|------|
| 对话模型必须和 Embedding 模型一致吗？ | **不需要**，它们是独立的两个模型 |
| 检索时的 Embedding 必须和存储时一致吗？ | **必须**，否则向量空间不一致 |
| 必须用 Ollama 吗？ | **不是**，支持 OpenAI、智谱、阿里等多种提供商 |
| 为什么结果像模型自己输出的？ | 检索结果被注入到提示词中，模型基于此生成自然回答 |

核心理解：**RAG = 检索 + 增强 + 生成**，用 Embedding 模型检索相关文档，用对话模型基于文档生成回答。
