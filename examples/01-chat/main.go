package main

import (
	"fmt"
	"log"

	"github.com/parakeet-nest/gollama"
	"github.com/parakeet-nest/gollama/enums/option"

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


	options := gollama.SetOptions(map[string]interface{}{
		option.Temperature: 0.5,
		option.RepeatLastN: 2,
		option.RepeatPenalty: 2.0,
	})

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
