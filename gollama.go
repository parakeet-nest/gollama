package gollama

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Answer struct {
	//Model   string  `json:"model"`
	Message Message `json:"message"`
	Done    bool    `json:"done"`
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

type Query struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Options  Options   `json:"options"`
	Stream   bool      `json:"stream"`

	Format    string `json:"format,omitempty"`
	KeepAlive bool   `json:"keep_alive,omitempty"`
	Raw       bool   `json:"raw,omitempty"`
	//System    string `json:"system,omitempty"`
	//Template  string `json:"template,omitempty"`
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
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#request-22
type Query4Embedding struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
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

	resp, err := http.Post(ollamaUrl+"/api/embeddings", "application/json; charset=utf-8", bytes.NewBuffer(jsonData))
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
//   - []llm.VectorRecord: a slice of vector records that have a cosine distance similarity greater than or equal to the limit.
//   - error: an error if any occurred during the search.
func (mvs *MemoryVectorStore) SearchSimilarities(embeddingFromQuestion VectorRecord, limit float64) ([]VectorRecord, error) {
	// search similarities
	var records []VectorRecord

	for _, v := range mvs.Records {
		distance := CosineDistance(embeddingFromQuestion.Embedding, v.Embedding)
		if distance >= limit {
			records = append(records, v)
		}
	}

	return records, nil
}

// === Similarities ===

// --- Jaccard index ---

// check if an item is a part of a set
func contains(set []string, element string) bool {
    for _, s := range set {
        if s == element {
            return true
        }
    }
    return false
}

// https://en.wikipedia.org/wiki/Jaccard_index
func JaccardSimilarityCoeff(set1, set2 []string) float64 {
    intersection := 0
    union := len(set1) + len(set2) - intersection

    for _, element := range set1 {
        if contains(set2, element) {
            intersection++
        }
    }

    return float64(intersection) / float64(union)
}

// --- Levenshtein distance ---

func min(a, b, c int) int {
    if a <= b && a <= c {
        return a
    } else if b <= a && b <= c {
        return b
    } else {
        return c
    }
}

func LevenshteinDistance(str1, str2 string) int {
    m := make([][]int, len(str1)+1)
    for i := range m {
        m[i] = make([]int, len(str2)+1)
    }

    for i := 0; i <= len(str1); i++ {
        for j := 0; j <= len(str2); j++ {
            if i == 0 {
                m[i][j] = j
            } else if j == 0 {
                m[i][j] = i
            } else if str1[i-1] == str2[j-1] {
                m[i][j] = m[i-1][j-1]
            } else {
                m[i][j] = 1 + min(m[i-1][j], m[i][j-1], m[i-1][j-1])
            }
        }
    }

    return m[len(str1)][len(str2)]
}


// === Chat Completion ===
func Chat(url string, query Query) (Answer, error) {

	query.Stream = false

	jsonQuery, err := json.Marshal(query)
	if err != nil {
		return Answer{}, err
	}

	resp, err := http.Post(url+"/api/chat", "application/json; charset=utf-8", bytes.NewBuffer(jsonQuery))
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

	resp, err := http.Post(url+"/api/chat", "application/json; charset=utf-8", bytes.NewBuffer(jsonQuery))
	if err != nil {
		return Answer{}, err
	}

	reader := bufio.NewReader(resp.Body)

	var fullAnswer Answer
	var answer Answer
	for {

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return Answer{}, err
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
	return fullAnswer, nil

}
