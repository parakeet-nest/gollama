package main

import (
	"fmt"
	"log"

	"github.com/parakeet-nest/gollama"
)

func main() {
	ollamaUrl := "http://localhost:11434"
	// if working from a container
	//ollamaUrl := "http://host.docker.internal:11434"
	model := "deepseek-coder"

	systemContent := `You are an expert in computer programming.
	Please make friendly answer for the noobs.
	Add source code examples if you can.`

	userContent := `I need a clear explanation regarding the following question:
	Can you create a "hello world" program in Golang?
	And, please, be structured with bullet points`

	options := gollama.Options{
		Temperature: 0.5, // default (0.8)
		RepeatLastN: 2,   // default (64) the default value will "freeze" deepseek-coder
		//Seed:        0, // default (0)
		RepeatPenalty: 2.0, // default (1.1)
		//Stop:        []string{},
	}

	query := gollama.Query{
		Model: model,
		Messages: []gollama.Message{
			{Role: "system", Content: systemContent},
			{Role: "user", Content: userContent},
		},
		Options: options,
	}

	answer, err := gollama.Chat(ollamaUrl, query)
	if err != nil {
		log.Fatal("😡:", err)
	}
	fmt.Println(answer.Message.Content)
}
