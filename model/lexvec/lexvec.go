// Copyright © 2017 Makoto Ito
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package lexvec

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/word-embedding/corpus"
	"github.com/ynqa/word-embedding/model"
)

// Lexvec stores the configs for Lexvec models.
type Lexvec struct {
	*model.Config
	*corpus.LexvecCorpus

	// hyper parameters.
	batchSize          int
	subSampleThreshold float64
	subSamples         []float64

	// words' vector.
	vector []float64

	// manage learning rate.
	currentlr        float64
	trained          chan struct{}
	trainedWordCount int

	// data size per thread.
	indexPerThread []int

	// progress bar.
	progress *pb.ProgressBar
}

// NewLexvec create *Lexvec.
func NewLexvec(f io.ReadCloser, config *model.Config, batchSize int, subSampleThreshold float64) *Lexvec {
	c := corpus.NewLexvecCorpus(f, config.ToLower, config.MinCount, config.Window)
	lexvec := &Lexvec{
		Config:       config,
		LexvecCorpus: c,

		batchSize:          batchSize,
		subSampleThreshold: subSampleThreshold,

		currentlr: config.Initlr,
		trained:   make(chan struct{}),
	}
	lexvec.initialize()
	return lexvec
}

func (l *Lexvec) initialize() {
	// Store subsample before training.
	l.subSamples = make([]float64, l.Corpus.Size())
	for i := 0; i < l.Corpus.Size(); i++ {
		z := 1. - math.Sqrt(l.subSampleThreshold/float64(l.IDFreq(i)))
		if z < 0 {
			z = 0
		}
		l.subSamples[i] = z
	}

	// Initialize word vector.
	vectorSize := l.Corpus.Size() * l.Config.Dimension
	l.vector = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		l.vector[i] = (rand.Float64() - 0.5) / float64(l.Config.Dimension)
	}
}

// Train trains words' vector on corpus.
func (l *Lexvec) Train() error {
	document := l.Document()
	documentSize := len(document)
	if documentSize <= 0 {
		return errors.New("No words for training")
	}

	l.indexPerThread = model.IndexPerThread(l.Config.ThreadSize, documentSize)

	for i := 1; i < l.Config.Iteration; i++ {
		if l.Config.Verbose {
			fmt.Printf("%d-th:\n", i)
			l.progress = pb.New(documentSize).SetWidth(80)
			l.progress.Start()
		}
		go l.observeLearningRate(i)

		semaphore := make(chan struct{}, l.Config.ThreadSize)
		waitGroup := &sync.WaitGroup{}

		for j := 0; j < l.Config.ThreadSize; j++ {
			waitGroup.Add(1)
			go l.trainPerThread(document, waitGroup, semaphore)
		}
		waitGroup.Wait()

		if l.Config.Verbose {
			l.progress.Finish()
		}
	}
	return nil
}

func (l *Lexvec) trainPerThread(document []int, waitGroup *sync.WaitGroup, semaphore chan struct{}) {
	defer func() {
		waitGroup.Done()
		<-semaphore
	}()

	for i, d := range document {
		if l.Config.Verbose {
			l.progress.Increment()
		}

		r := rand.Float64()
		p := l.subSamples[d]
		if p < r {
			continue
		}
		_ = fmt.Sprintln(i, d)
		l.trained <- struct{}{}
	}
}

func (l *Lexvec) observeLearningRate(iteration int) {
	for range l.trained {
		l.trainedWordCount++
		if l.trainedWordCount%l.batchSize == 0 {
			l.currentlr = l.Config.Initlr *
				(1. - float64(l.trainedWordCount)/
					(float64(l.Corpus.TotalFreq())-float64(iteration)))
		}
	}
}
