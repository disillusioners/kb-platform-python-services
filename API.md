# Python Services - Core gRPC Specification

## Overview

This service interacts with the Go API Gateway via **gRPC**. The Proto definitions are the source of truth for this interface.

**Repository**: `kb-platform-proto` (Shared definitions)

## gRPC Service Definition

### Service Name: `KBPlatformService`

```protobuf
syntax = "proto3";

package kbplatform.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

option go_package = "github.com/your-org/kb-platform-proto/gen/go/kbplatform/v1";

service KBPlatformService {
  // Query / RAG
  // Server-side streaming for RAG response chunks
  rpc QueryStream (QueryRequest) returns (stream QueryResponse);

  // Document Management
  rpc GetDocument (GetDocumentRequest) returns (Document);
  rpc DeleteDocumentVectors (DeleteDocumentVectorsRequest) returns (google.protobuf.Empty);

  // Conversation Management
  rpc GetConversation (GetConversationRequest) returns (Conversation);
  rpc GetConversationMessages (GetConversationMessagesRequest) returns (GetConversationMessagesResponse);
  rpc SaveMessage (SaveMessageRequest) returns (Message);
}
```

## Messages Requests & Responses

### Query

```protobuf
message QueryRequest {
  string query = 1;
  string conversation_id = 2; // UUID
  int32 top_k = 3;            // Default: 5
}

message QueryResponse {
  oneof event {
    QueryEventStart start = 1;
    QueryEventChunk chunk = 2;
    QueryEventEnd end = 3;
    QueryEventError error = 4;
  }
}

message QueryEventStart {
  string request_id = 1;
}

message QueryEventChunk {
  string content = 1;
}

message QueryEventEnd {
  string request_id = 1;
}

message QueryEventError {
  string code = 1;
  string message = 2;
}
```

### Documents

```protobuf
message GetDocumentRequest {
  string document_id = 1; // UUID
}

message DeleteDocumentVectorsRequest {
  string document_id = 1; // UUID
}

message Document {
  string id = 1;
  string filename = 2;
  int64 file_size = 3;
  string status = 4; // pending, indexing, complete, failed
  google.protobuf.Timestamp created_at = 5;
  google.protobuf.Timestamp indexed_at = 6;
  string error_message = 7;
  map<string, string> metadata = 8;
}
```

### Conversations

```protobuf
message GetConversationRequest {
  string conversation_id = 1; // UUID
}

message Conversation {
  string id = 1;
  google.protobuf.Timestamp created_at = 2;
  google.protobuf.Timestamp updated_at = 3;
  int32 message_count = 4;
}

message GetConversationMessagesRequest {
  string conversation_id = 1; // UUID
}

message GetConversationMessagesResponse {
  repeated Message messages = 1;
}

message SaveMessageRequest {
  string conversation_id = 1;
  string role = 2; // user, assistant
  string content = 3;
  map<string, string> metadata = 4;
}

message Message {
  string id = 1;
  string conversation_id = 2;
  string role = 3;
  string content = 4;
  google.protobuf.Timestamp timestamp = 5;
  map<string, string> metadata = 6;
}
```

## Health Checks

Standard gRPC Health Checking Protocol should be implemented.
- Service: `grpc.health.v1.Health`
- Methods: `Check`, `Watch`

