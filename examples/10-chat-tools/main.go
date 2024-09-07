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
	model := "allenporter/xlam:1b"

	toolsList := []gollama.Tool{
		{
			Type: "function",
			Function: gollama.Function{
				Name:        "hello",
				Description: "Say hello to a given person with his name",
				Parameters: gollama.Parameters{
					Type: "object",
					Properties: map[string]gollama.Property{
						"name": {
							Type:        "string",
							Description: "The name of the person",
						},
					},
					Required: []string{"name"},
				},
			},
		},
		{
			Type: "function",
			Function: gollama.Function{
				Name:        "addNumbers",
				Description: "Make an addition of the two given numbers",
				Parameters: gollama.Parameters{
					Type: "object",
					Properties: map[string]gollama.Property{
						"a": {
							Type:        "number",
							Description: "first operand",
						},
						"b": {
							Type:        "number",
							Description: "second operand",
						},
					},
					Required: []string{"a", "b"},
				},
			},
		},
	}

	messages := []gollama.Message{
		{Role: "user", Content: `say "hello" to Bob`},
	}

	options := gollama.Options{
		Temperature:   0.0,
		RepeatLastN:   2,
		RepeatPenalty: 2.0,
	}

	query := gollama.Query{
		Model:    model,
		Messages: messages,
		Tools:    toolsList,
		Options:  options,
		Format:   "json",
	}

	answer, err := gollama.Chat(ollamaUrl, query)
	if err != nil {
		log.Fatal("😡:", err)
	}

	// TODO: test the number of tools before
	result, err := answer.Message.ToolCalls[0].Function.ToJSONString()
	if err != nil {
		log.Fatal("😡:", err)
	}
	fmt.Println(result)

	messages = []gollama.Message{
		{Role: "user", Content: `add 2 and 40`},
	}

	query = gollama.Query{
		Model:    model,
		Messages: messages,
		Tools:    toolsList,
		Options:  options,
		Format:   "json",
	}

	answer, err = gollama.Chat(ollamaUrl, query)
	if err != nil {
		log.Fatal("😡:", err)
	}

	// TODO: test the number of tools before
	result, err = answer.Message.ToolCalls[0].Function.ToJSONString()
	if err != nil {
		log.Fatal("😡:", err)
	}
	fmt.Println(result)

}
