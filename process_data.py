def process_data(data,spacy_pipeline):
    feature_data = {key: [section.copy() for section in value] for key, value in data.items()}
    print("\n--- Running Step 1: Unified Feature Extraction ---")
    for article_id,sections in feature_data.items():
        article_noun_chunks = []
        for section in sections:
            orginal_text = section["text"]
            section["original_text"] = orginal_text
            doc = spacy_pipeline(orginal_text)
            cleaned_text = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
            section["cleaned_text"] = "".join(cleaned_text)

            for chunk in doc.noun_chunks:
                article_noun_chunks.append(chunk)
        feature_data[article_id].append({
            'section': 'extracted_features',
            'noun_chunks': list(set(article_noun_chunks))
        })
    print("--- Finished Step 1 ---")
    return feature_data
