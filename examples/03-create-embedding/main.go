package main

import (
	"fmt"
	"log"
	"github.com/parakeet-nest/gollama"
)

func main() {
	ollamaUrl := "http://localhost:11434"
	model := "all-minilm"


	content := `The best pizza of the world is the pineapple pizza`


	query := gollama.Query4Embedding{
		Model: model,
		Prompt: content,
	}

	vector, err := gollama.CreateEmbedding(ollamaUrl, query, "000")

	if err != nil {
		log.Fatal("😡:", err)
	}
	fmt.Println(vector)
	//-0.28555548191070557 0.11453928798437119 0.04224501550197601

}
