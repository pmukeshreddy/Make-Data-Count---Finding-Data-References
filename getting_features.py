import re
def genrate_final_canidates(data,ner_pipeline):
    final_canidates_by_article = {}
    regex_patterns = [
        r'\b[A-Z]+-[A-Z]+-\d+\b',
        r'\bGSE\d+\b',
        r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,}\b'
    ]
    combined_regex = re.compile('|'.join(regex_patterns))
    print("\n--- Running Step 2: Final Candidate Generation ---")
    for article_id, sections in data.items():
        unique_candidates = set()
        #first thing
        for section in sections:
            if section.get("section") == "extracted_features":
                unique_candidates.update(section.get('noun_chunks', []))
                break

        for section in sections:
            if section.get("section") == "extracted_features":
                continue
            text_to_process = section.get('original_text', section.get('text', ''))
            #second thing
            for match in re.finditer(combined_regex,text_to_process):
                unique_candidates.add(match.group(0).strip())
            #thrid thing
            ner_results = ner_pipeline(text_to_process)
            for entity in ner_results:
                unique_candidates.add(entity["word"].strip())
        final_canidates_by_article[article_id] = list(unique_candidates)
    print("--- Finished Step 2 ---")
    return final_canidates_by_article
