package gollama

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"text/template"
)

type FunctionTool struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"` // used for the ToolCalls list
}

func (ft *FunctionTool) ToJSONString() (string, error) {
	// Marshal the data into JSON
	jsonBytes, err := json.Marshal(ft)
	if err != nil {
		return "", err
	}
	// Convert JSON bytes to string
	jsonString := string(jsonBytes)
	return jsonString, nil
}

type Message struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ToolCalls []struct {
		Function FunctionTool `json:"function"`
	} `json:"tool_calls"`
}

func (m *Message) ToolCallsToJSONString() (string, error) {
	// Marshal the data into JSON
	jsonBytes, err := json.Marshal(m.ToolCalls)
	if err != nil {
		return "", err
	}
	// Convert JSON bytes to string
	jsonString := string(jsonBytes)
	return jsonString, nil
}

func (m *Message) FirstToolCallToJSONString() (string, error) {
	// Marshal the data into JSON
	jsonBytes, err := json.Marshal(m.ToolCalls[0])
	if err != nil {
		return "", err
	}
	// Convert JSON bytes to string
	jsonString := string(jsonBytes)
	return jsonString, nil
}

// Answer
type Answer struct {
	Model   string  `json:"model"`
	Message Message `json:"message"`
	Done    bool    `json:"done"`
}

func (answer *Answer) ToJsonString() string {
	// for the verbose mode
	// Marshal the answer into JSON
	jsonBytes, err := json.MarshalIndent(answer, "", "  ")

	if err != nil {
		return `{"error":"` + err.Error() + `"}`
	}
	// Convert JSON bytes to string
	jsonString := string(jsonBytes)
	return jsonString
}

type Options struct {
	RepeatLastN   int      `json:"repeat_last_n,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	Seed          int      `json:"seed,omitempty"`
	RepeatPenalty float64  `json:"repeat_penalty,omitempty"`
	Stop          []string `json:"stop,omitempty"`

	NumKeep          int     `json:"num_keep,omitempty"`
	NumPredict       int     `json:"num_predict,omitempty"`
	TopK             int     `json:"top_k,omitempty"`
	TopP             float64 `json:"top_p,omitempty"`
	TFSZ             float64 `json:"tfs_z,omitempty"`
	TypicalP         float64 `json:"typical_p,omitempty"`
	PresencePenalty  float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`
	Mirostat         int     `json:"mirostat,omitempty"`
	MirostatTau      float64 `json:"mirostat_tau,omitempty"`
	MirostatEta      float64 `json:"mirostat_eta,omitempty"`
	PenalizeNewline  bool    `json:"penalize_newline,omitempty"`
}

// Tools
type Property struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

type Parameters struct {
	Type       string              `json:"type"`
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required"`
}

type Function struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Query
type Query struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"` // For Chat Completion
	Options  Options   `json:"options"`
	Stream   bool      `json:"stream"`
	Tools    []Tool    `json:"tools"`

	Format    string `json:"format,omitempty"` // https://github.com/ollama/ollama/blob/main/docs/api.md#request-json-mode
	KeepAlive bool   `json:"keep_alive,omitempty"`
	Raw       bool   `json:"raw,omitempty"`
	System    string `json:"system,omitempty"`
	Template  string `json:"template,omitempty"`

	TokenHeaderName  string
	TokenHeaderValue string
}

func (query *Query) ToJsonString() string {
	// for the verbose mode
	// Marshal the answer into JSON
	jsonBytes, err := json.MarshalIndent(query, "", "  ")

	if err != nil {
		return `{"error":"` + err.Error() + `"}`
	}
	// Convert JSON bytes to string
	jsonString := string(jsonBytes)
	return jsonString
}

// === Cosine distance ===
func dotProduct(v1 []float64, v2 []float64) float64 {
	// Calculate the dot product of two vectors
	sum := 0.0
	for i := range v1 {
		sum += v1[i] * v2[i]
	}
	return sum
}

func CosineDistance(v1, v2 []float64) float64 {
	// Calculate the cosine distance between two vectors
	product := dotProduct(v1, v2)
	norm1 := math.Sqrt(dotProduct(v1, v1))
	norm2 := math.Sqrt(dotProduct(v2, v2))
	if norm1 <= 0.0 || norm2 <= 0.0 {
		// Handle potential division by zero
		return 0.0
	}
	return product / (norm1 * norm2)
}

// === Embeddings ===
type VectorRecord struct {
	Id        string    `json:"id"`
	Prompt    string    `json:"prompt"`
	Embedding []float64 `json:"embedding"`

	CosineDistance float64

	Reference string `json:"reference"`
	MetaData  string `json:"metaData"`
	Text      string `json:"text"`
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#request-22
type Query4Embedding struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`

	TokenHeaderName  string
	TokenHeaderValue string
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// Create embedding
func CreateEmbedding(ollamaUrl string, query Query4Embedding, id string) (VectorRecord, error) {
	jsonData, err := json.Marshal(query)
	if err != nil {
		return VectorRecord{}, err
	}

	req, err := http.NewRequest(http.MethodPost, ollamaUrl+"/api/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return VectorRecord{}, err
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	if query.TokenHeaderName != "" && query.TokenHeaderValue != "" {
		req.Header.Set(query.TokenHeaderName, query.TokenHeaderValue)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return VectorRecord{}, err
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return VectorRecord{}, errors.New("Error: status code: " + resp.Status)
	}
	body, err := io.ReadAll(resp.Body)

	if err != nil {
		return VectorRecord{}, err
	}

	var answer EmbeddingResponse
	err = json.Unmarshal([]byte(string(body)), &answer)
	if err != nil {
		return VectorRecord{}, err
	}

	vectorRecord := VectorRecord{
		Prompt:    query.Prompt,
		Embedding: answer.Embedding,
		Id:        id,
	}

	return vectorRecord, nil
}

// === Vector Store

type MemoryVectorStore struct {
	Records map[string]VectorRecord
}

func (mvs *MemoryVectorStore) Get(id string) (VectorRecord, error) {
	return mvs.Records[id], nil
}

func (mvs *MemoryVectorStore) GetAll() ([]VectorRecord, error) {
	var records []VectorRecord
	for _, record := range mvs.Records {
		records = append(records, record)
	}
	return records, nil
}

func (mvs *MemoryVectorStore) Save(vectorRecord VectorRecord) (VectorRecord, error) {
	mvs.Records[vectorRecord.Id] = vectorRecord
	return vectorRecord, nil
}

// SearchSimilarities searches for vector records in the MemoryVectorStore that have a cosine distance similarity greater than or equal to the given limit.
//
// Parameters:
//   - embeddingFromQuestion: the vector record to compare similarities with.
//   - limit: the minimum cosine distance similarity threshold.
//
// Returns:
//   - VectorRecord: a slice of vector records that have a cosine distance similarity greater than or equal to the limit.
//   - error: an error if any occurred during the search.
func (mvs *MemoryVectorStore) SearchSimilarities(embeddingFromQuestion VectorRecord, limit float64) ([]VectorRecord, error) {
	// search similarities
	var records []VectorRecord

	for _, v := range mvs.Records {
		distance := CosineDistance(embeddingFromQuestion.Embedding, v.Embedding)
		if distance >= limit {
			v.CosineDistance = distance
			records = append(records, v)
		}
	}

	return records, nil
}

// SearchTopNSimilarities searches for the top N similar vector records based on the given embedding from a question.
// It returns a slice of vector records and an error if any.
// The limit parameter specifies the minimum similarity score for a record to be considered similar.
// The max parameter specifies the maximum number of vector records to return.
func (mvs *MemoryVectorStore) SearchTopNSimilarities(embeddingFromQuestion VectorRecord, limit float64, max int) ([]VectorRecord, error) {
	records, err := mvs.SearchSimilarities(embeddingFromQuestion, limit)
	if err != nil {
		return nil, err
	}
	return getTopNVectorRecords(records, max), nil
}

func getTopNVectorRecords(records []VectorRecord, max int) []VectorRecord {
	// Sort the records slice in descending order based on CosineDistance
	sort.Slice(records, func(i, j int) bool {
		return records[i].CosineDistance > records[j].CosineDistance
	})

	// Return the first max records or all if less than three
	if len(records) < max {
		return records
	}
	return records[:max]
}

// === Chat Completion ===
func Chat(url string, query Query) (Answer, error) {

	query.Stream = false

	jsonQuery, err := json.Marshal(query)
	if err != nil {
		return Answer{}, err
	}

	req, err := http.NewRequest(http.MethodPost, url+"/api/chat", bytes.NewBuffer(jsonQuery))
	if err != nil {
		return Answer{}, err
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	if query.TokenHeaderName != "" && query.TokenHeaderValue != "" {
		req.Header.Set(query.TokenHeaderName, query.TokenHeaderValue)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return Answer{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Answer{}, errors.New("Error: status code: " + resp.Status)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Answer{}, err
	}

	var answer Answer
	err = json.Unmarshal(body, &answer)

	if err != nil {
		return Answer{}, err
	}

	return answer, nil

}

func ChatStream(url string, query Query, onChunk func(Answer) error) (Answer, error) {

	query.Stream = true

	jsonQuery, err := json.Marshal(query)
	if err != nil {
		return Answer{}, err
	}

	req, err := http.NewRequest(http.MethodPost, url+"/api/chat", bytes.NewBuffer(jsonQuery))
	if err != nil {
		return Answer{}, err
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	if query.TokenHeaderName != "" && query.TokenHeaderValue != "" {
		req.Header.Set(query.TokenHeaderName, query.TokenHeaderValue)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return Answer{}, err
	}
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)

	var fullAnswer Answer
	var answer Answer
	for {

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			// we need to create a new error because
			// because, even if the status is not ok (ex 401 Unauthorized)
			// the error == nil
			//return Answer{}, errors.New("Error: status code: " + resp.Status)
			return Answer{}, errors.New("Error: status code: " + resp.Status)
		}

		err = json.Unmarshal(line, &answer)
		if err != nil {
			onChunk(Answer{})
		}
		fullAnswer.Message.Content += answer.Message.Content
		err = onChunk(answer)

		// generate an error to stop the stream
		if err != nil {
			return Answer{}, err
		}
	}
	fullAnswer.Message.Role = answer.Message.Role
	//return fullAnswer, nil
	if resp.StatusCode != http.StatusOK {
		return Answer{}, errors.New("Error: status code: " + resp.Status)
	} else {
		return fullAnswer, nil
	}

}

// String Interpolation
func process(t *template.Template, vars interface{}) (string, error) {
	var tmplBytes bytes.Buffer

	err := t.Execute(&tmplBytes, vars)
	if err != nil {
		return "", err
	}
	return tmplBytes.String(), nil
}

func InterpolateString(str string, vars interface{}) (string, error) {

	tmpl, err := template.New("tmpl").Parse(str)

	if err != nil {
		return "", err
	}
	output, err := process(tmpl, vars)

	if err != nil {
		return "", err
	}
	return output, nil
}

// Splitter

// SplitTextWithDelimiter splits the given text using the specified delimiter and returns a slice of strings.
//
// Parameters:
//   - text: The text to be split.
//   - delimiter: The delimiter used to split the text.
//
// Returns:
//   - []string: A slice of strings containing the split parts of the text.
func SplitTextWithDelimiter(text string, delimiter string) []string {
	return strings.Split(text, delimiter)
}
