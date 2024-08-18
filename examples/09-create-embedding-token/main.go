package main

import (
	"fmt"
	"log"
	"os"

	"github.com/parakeet-nest/gollama"
)

func main() {
	ollamaUrl := "https://ollamak33g.eu.loclx.io"
	model := "all-minilm:33m"

	content := `The best pizza of the world is the pineapple pizza`

	query := gollama.Query4Embedding{
		Model:            model,
		Prompt:           content,
		TokenHeaderName:  "X-TOKEN",
		TokenHeaderValue: os.Getenv("OLLAMA_TOKEN"),
	}

	vector, err := gollama.CreateEmbedding(ollamaUrl, query, "000")

	if err != nil {
		log.Fatal("😡:", err)
	}
	fmt.Println(vector)
	//-0.28555548191070557 0.11453928798437119 0.04224501550197601

}
