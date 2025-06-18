def create_multitask_training(canidate_by_atricle,articles_data,train_lables_df,spacy_nlp):
    primary_label_mappig = {'Primary': 1, 'Secondary': 2}
    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE", "UNK"]
    pos_label_map = {tag:i for i,tag in enumerate(pos_tags)}
    training_smaples = []
    print("--- Preparing data for Multi-Task LLM Fine-Tuning ---")
    for article_id,canidates in canidate_by_atricle.items():
        article_label_df = train_lables_df[train_lables_df["article_id"]==article_id]
        text_sections = articles_data.get(article_id,[])

        for canidate in canidates:
            sentence_context , first_word_pos = get_sentence_and_pos(canidate,text_sections,spacy_nlp)
            if not sentence_context:
                continue

            match = article_label_df[train_lables_df["dataset_id"]==canidate]
            primary_label = pos_label_map.get(match.iloc[0]["type"],0) if not match.empty else 0
            pos_label = pos_label_map.get(first_word_pos,pos_label_map["UNK"])

            training_samples.append({'text': sentence_context, 'primary_label': primary_label, 'pos_label': pos_label})
    print(f"Created {len(training_samples)} training samples.")
    return training_samples, len(pos_tags)
