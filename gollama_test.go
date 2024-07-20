package gollama

import (
	"log"
	"testing"
)

func TestEmbeddingCreation(t *testing.T) {
	ollamaUrl := "http://localhost:11434"
	model := "all-minilm"

	content := `The best pizza of the world is the pineapple pizza`

	query := Query4Embedding{
		Model:  model,
		Prompt: content,
	}

	vector, err := CreateEmbedding(ollamaUrl, query, "000")

	if err != nil {
		t.Fatal("😡:", err)
	}

	if vector.Embedding[0] == -0.28555548191070557 && vector.Embedding[1] == 0.11453928798437119 && vector.Embedding[2] == 0.04224501550197601 {
		log.Println("🙂", vector)
	} else {
		t.Fatal("😡:", err)
	}

}
