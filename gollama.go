package gollama

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Answer struct {
	//Model   string  `json:"model"` // 🤔
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
