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

package corpus

import (
	"io"

	"github.com/ynqa/word-embedding/corpus/co"
)

// LexvecCorpus stores corpus and MI between words.
type LexvecCorpus struct {
	*core
	costats map[uint64]float64
}

// NewLexvecCorpus creates *LexvecCorpus.
func NewLexvecCorpus(f io.ReadCloser, toLower bool, minCount, window int) *LexvecCorpus {
	lexvecCorpus := &LexvecCorpus{
		core:    newCore(),
		costats: make(map[uint64]float64),
	}
	lexvecCorpus.parse(f, toLower, minCount)
	lexvecCorpus.build(window)
	return lexvecCorpus
}

// Costats returns statistics map based on co-occurrence for words.
func (lc *LexvecCorpus) Costats() map[uint64]float64 {
	return lc.costats
}

func (lc *LexvecCorpus) build(window int) {
	for i := 0; i < len(lc.document); i++ {
		// TODO: random shrinkage for window like word2vec.
		for j := i + 1; j <= i+window; j++ {
			if j >= len(lc.document) {
				continue
			}
			lc.costats[co.EncodeBigram(uint64(lc.document[i]), uint64(lc.document[j]))] += 1.
			lc.costats[co.EncodeBigram(uint64(lc.document[j]), uint64(lc.document[i]))] += 1.
		}
	}
}
