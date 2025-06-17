def create_feature_dataset(candidates_by_article, articles_data, train_labels_df, model, tokenizer):
    all_known_datasets = set(train_labels_df['dataset_id'].unique())
    id_regex = re.compile(r"(\b[A-Z]+-[A-Z]+-\d+\b|\bGSE\d+\b)")
    label_map = {'Primary': 1, 'Secondary': 2}
    final_dataset = []
    for article_id , canidates in candidates_by_article.items():
        print(f"Processing {len(candidates)} candidates for article: {article_id}")
        article_labels_df =  train_labels_df[train_labels_df["article_id"]== article_id]
        text_sections = articles_data.get(article_id,[])
        for candiate in canidates:
            match =  article_labels_df[article_labels_df["dataset_id"]==candiate]
            if not match.empty:
                dataset_type = match.iloc[0].["type"]
                label = label_map.get(dataset_type,0)
            else:
                label = 0
            sentence_context = get_sentence_for_canidate(candiate,text_sections)
            if not sentence_context:
                continue
                
            encoded_input = tokenizer(sentence_context,returns="pt",truncation=True,max_length=512)

            with torch.no_grad():
                outputs = model(**encoded_input)
                token_embeddings = outputs.last_hidden_state

            candidate_token = tokenizer.tokenize(candiate)
            input_tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])

            start_index , end_index = -1 , -1
            for i in range(len(input_tokens)-len(candidate_token)+1):
                if input_tokens[i:i+len(candidate_token)] == candidate_token:
                    start_index = i
                    end_index = i + len(candidate_token)
                    
            if start_index != -1:
                canidate_embeddings = token_embeddings[0,start_index:end_index,:].mean(dim=0).numpy()

            else:
                candidate_embedding = np.zeros(model.config.hidden_size)
            feature_dict = {
                'article_id': article_id,
                'candidate_text': candidate,
                'label': label, # This is now our multi-class label (0, 1, or 2)
                'sciBERT_embedding': candidate_embedding,
                'is_all_caps': 1 if candidate.isupper() and len(candidate) > 1 else 0,
                'contains_digits': 1 if any(char.isdigit() for char in candidate) else 0,
                'candidate_length_tokens': len(candidate.split()),
                'candidate_length_chars': len(candidate),
                'is_standalone_id': 1 if id_regex.fullmatch(candidate) else 0,
                'is_in_training_gazetteer': 1 if candidate in all_known_datasets else 0
            }
            
            final_dataset.append(feature_dict)
    print("\n--- Finished Step 3 ---")
    return final_dataset
