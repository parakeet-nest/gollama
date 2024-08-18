package main

import (
	"fmt"
	"log"
	"strconv"

	"github.com/parakeet-nest/gollama"
)

var docs = []string{
	`Michael Burnham is the main character on the Star Trek series, Discovery.  
	She's a human raised on the logical planet Vulcan by Spock's father.  
	Burnham is intelligent and struggles to balance her human emotions with Vulcan logic.  
	She's become a Starfleet captain known for her determination and problem-solving skills.
	Originally played by actress Sonequa Martin-Green`,

	`James T. Kirk, also known as Captain Kirk, is a fictional character from the Star Trek franchise.  
	He's the iconic captain of the starship USS Enterprise, 
	boldly exploring the galaxy with his crew.  
	Originally played by actor William Shatner, 
	Kirk has appeared in TV series, movies, and other media.`,

	`Jean-Luc Picard is a fictional character in the Star Trek franchise.
	He's most famous for being the captain of the USS Enterprise-D,
	a starship exploring the galaxy in the 24th century.
	Picard is known for his diplomacy, intelligence, and strong moral compass.
	He's been portrayed by actor Patrick Stewart.`,

	`Lieutenant Philippe Charrière, known as the **Silent Sentinel** of the USS Discovery, 
	is the enigmatic programming genius whose codes safeguard the ship's secrets and operations. 
	His swift problem-solving skills are as legendary as the mysterious aura that surrounds him. 
	Charrière, a man of few words, speaks the language of machines with unrivaled fluency, 
	making him the crew's unsung guardian in the cosmos. His best friend is Spiderman from the Marvel Cinematic Universe.`,
}

func main() {
	ollamaUrl := "http://localhost:11434"
	//embeddingsModel := "all-minilm:22m"
	//embeddingsModel := "all-minilm:33m"
	//embeddingsModel := "qwen2:0.5b"
	embeddingsModel := "qwen2:1.5b"


	store := gollama.MemoryVectorStore{
		Records: make(map[string]gollama.VectorRecord),
	}

	// Create embeddings from documents and save them in the store
	for idx, doc := range docs {
		fmt.Println("Creating embedding from document ", idx)
		embedding, err := gollama.CreateEmbedding(
			ollamaUrl,
			gollama.Query4Embedding{
				Model:  embeddingsModel,
				Prompt: doc,
			},
			strconv.Itoa(idx),
		)
		if err != nil {
			fmt.Println("😡:", err)
		} else {
			store.Save(embedding)
		}
	}

	userContent := `Who is Philippe Charrière and what spaceship does he work on?`
	//userContent := `Who is Philippe Charrière?`

	// Create an embedding from the question
	embeddingFromQuestion, err := gollama.CreateEmbedding(
		ollamaUrl,
		gollama.Query4Embedding{
			Model:  embeddingsModel,
			Prompt: userContent,
		},
		"question",
	)
	if err != nil {
		log.Fatalln("😡:", err)
	}
	fmt.Println("🔎 searching for similarity...")

	// decrease the limit with all-minilm:22m & 33m -> 0.4
	// decrease the limit with qwen2:0.5b & 1.5b -> 0.6
	
	//similarities, err := store.SearchSimilarities(embeddingFromQuestion, 0.6)

	similarities, err := store.SearchTopNSimilarities(embeddingFromQuestion, 0.3, 3)
	/*
	- put the limit to 0.5 
	- and get only the last n results
	*/

	if err != nil {
		log.Fatalln("😡:", err)
	}

	fmt.Println("🔎 nb:", len(similarities))
	
	for _, similarity := range similarities {
		fmt.Println(similarity.Prompt)
	}

}
