from transformers import MarianMTModel, MarianTokenizer

class BackTranslator:
    def __init__(self):
        # Load FR->EN model
        self.fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
        self.fr_en_tokenizer = MarianTokenizer.from_pretrained(self.fr_en_model_name)
        self.fr_en_model = MarianMTModel.from_pretrained(self.fr_en_model_name)
        
        # Load EN->FR model
        self.en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
        self.en_fr_tokenizer = MarianTokenizer.from_pretrained(self.en_fr_model_name)
        self.en_fr_model = MarianMTModel.from_pretrained(self.en_fr_model_name)
    
    def back_translate(self, french_text, max_length=256):
        """Perform FR->EN->FR back-translation with chunking."""
        try:
            # Check if text is too short to need chunking
            fr_token_count = len(self.fr_en_tokenizer.encode(french_text))
            
            if fr_token_count <= max_length:
                # Use simple translation for short texts
                return self._simple_translate(french_text, max_length)
            else:
                # Use chunking for long texts
                return self._chunked_translate(french_text, max_length)
                
        except Exception as e:
            print(f"Back-translation failed: {e}")
            return french_text  # Return original text if everything fails
    
    def back_translate_batch(self, french_texts, max_length=400):
        """Back-translate a batch of texts."""
        results = []
        for i, text in enumerate(french_texts):
            print(f"Processing text {i+1}/{len(french_texts)}")
            result = self.back_translate(text, max_length)
            results.append(result)
        return results

    def _split_text_into_chunks(self, text, tokenizer, max_length=400):
        """Split text into chunks that fit within the model's token limit."""
        # Split by sentences first (better semantic boundaries)
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Test if adding this sentence would exceed the limit
            test_chunk = current_chunk + sentence + '. ' if current_chunk else sentence + '. '
            
            # Check token count
            token_count = len(tokenizer.encode(test_chunk))
            
            if token_count > max_length and current_chunk:
                # Current chunk is full, save it and start new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
            else:
                current_chunk = test_chunk
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _translate_chunks(self, chunks, source_tokenizer, source_model, target_tokenizer, max_length=400):
        """Translate a list of text chunks."""
        translated_chunks = []
        
        for chunk in chunks:
            try:
                # Tokenize the chunk
                batch = source_tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                
                # Translate with improved parameters
                translated = source_model.generate(
                    **batch,
                    max_length=max_length,  # Ensure the output does not exceed the maximum length
                    num_beams=4,  # Use beam search for better translation quality
                    early_stopping=True,  # Stop early if the translation is complete
                    no_repeat_ngram_size=3,  # Avoid repeating n-grams in the translation
                    do_sample=False,  # Enable sampling for more diverse translations
                    temperature=1.0  # Temperature for sampling (1.0 is neutral, >1.0 is more random)
                )

                # Decode the translation
                result = target_tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_chunks.append(result)
                
            except Exception as e:
                print(f"Warning: Failed to translate chunk: {e}")
                # Return original chunk if translation fails
                translated_chunks.append(chunk)
        
        return translated_chunks
    
    
    def _simple_translate(self, french_text, max_length):
        """Simple translation without chunking for shorter texts."""
        # FR->EN
        batch_fr_en = self.fr_en_tokenizer([french_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        english_tokens = self.fr_en_model.generate(
            **batch_fr_en,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        english = self.en_fr_tokenizer.decode(english_tokens[0], skip_special_tokens=True)
        
        # EN->FR
        batch_en_fr = self.en_fr_tokenizer([english], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        french_tokens = self.en_fr_model.generate(
            **batch_en_fr,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        back_to_french = self.fr_en_tokenizer.decode(french_tokens[0], skip_special_tokens=True)
        
        return back_to_french
    
    def _chunked_translate(self, french_text, max_length):
        """Chunked translation for longer texts."""
        # Step 1: Split French text into chunks
        fr_chunks = self._split_text_into_chunks(french_text, self.fr_en_tokenizer, max_length)
        print(f"Split text into {len(fr_chunks)} chunks for translation")
        
        # Step 2: Translate FR->EN chunks
        en_chunks = self._translate_chunks(fr_chunks, self.fr_en_tokenizer, self.fr_en_model, self.en_fr_tokenizer, max_length)
        
        # Step 3: Join English chunks
        english_text = ' '.join(en_chunks)
        
        # Step 4: Split English text into chunks (might be different from French chunks)
        en_chunks_for_back = self._split_text_into_chunks(english_text, self.en_fr_tokenizer, max_length)
        
        # Step 5: Translate EN->FR chunks
        fr_back_chunks = self._translate_chunks(en_chunks_for_back, self.en_fr_tokenizer, self.en_fr_model, self.fr_en_tokenizer, max_length)
        
        # Step 6: Join final French chunks
        back_to_french = ' '.join(fr_back_chunks)
        
        return back_to_french