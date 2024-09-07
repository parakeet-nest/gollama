# GoLlama

**GoLlama** is a very simple library to interact with the Ollama API. (It's a wrapper around the Ollama API + some helper functions)

> GoLlama was developed for the preparation of a presentation at DevFest Toulouse 2024. 

## Features

- Chat completion
- Embedding
- Tools

```mermaid
classDiagram
    class FunctionTool {
        +string Name
        +map[string, interface> Arguments
        +ToJSONString() string
    }

    class Message {
        +string Role
        +string Content
        +ToolCalls[]
        +ToolCallsToJSONString() string
        +FirstToolCallToJSONString() string
    }

    class Answer {
        +string Model
        +Message Message
        +bool Done
        +ToJsonString() string
    }

    class Options {
        +int RepeatLastN
        +float64 Temperature
        +int Seed
        +float64 RepeatPenalty
        +string[] Stop
        +int NumKeep
        +int NumPredict
        +int TopK
        +float64 TopP
        +float64 TFSZ
        +float64 TypicalP
        +float64 PresencePenalty
        +float64 FrequencyPenalty
        +int Mirostat
        +float64 MirostatTau
        +float64 MirostatEta
        +bool PenalizeNewline
    }

    class Property {
        +string Type
        +string Description
    }

    class Parameters {
        +string Type
        +map[string, Property> Properties
        +string[] Required
    }

    class Function {
        +string Name
        +string Description
        +Parameters Parameters
    }

    class Tool {
        +string Type
        +Function Function
    }

    class Query {
        +string Model
        +Message[] Messages
        +Options Options
        +bool Stream
        +Tool[] Tools
        +string Format
        +bool KeepAlive
        +bool Raw
        +string System
        +string Template
        +string TokenHeaderName
        +string TokenHeaderValue
        +ToJsonString() string
    }

    class VectorRecord {
        +string Id
        +string Prompt
        +float64[] Embedding
        +float64 CosineDistance
        +string Reference
        +string MetaData
        +string Text
    }

    class Query4Embedding {
        +string Prompt
        +string Model
        +string TokenHeaderName
        +string TokenHeaderValue
    }

    class EmbeddingResponse {
        +float64[] Embedding
    }

    class MemoryVectorStore {
        +map[string, VectorRecord> Records
        +Get(string) VectorRecord
        +GetAll() VectorRecord[]
        +Save(VectorRecord) VectorRecord
        +SearchSimilarities(VectorRecord, float64) VectorRecord[]
        +SearchTopNSimilarities(VectorRecord, float64, int) VectorRecord[]
    }

    FunctionTool --> Message
    Message --> Tool
    Answer --> Message
    Query --> Message
    Query --> Options
    Query --> Tool
    Tool --> Function
    Function --> Parameters
    Parameters --> Property
    MemoryVectorStore --> VectorRecord
    VectorRecord --> EmbeddingResponse
    Query4Embedding --> VectorRecord

```