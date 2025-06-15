import xml.etree.ElementTree as ET
from pathlib import Path
import json

def extract_structured_text_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        structured_text = []

        article_text_element = root.find('.//article-title')

        if article_text_element is not None and article_text_element.text:
            structured_text.append({
                'section': 'Title',
                'text': article_text_element.text.strip()
            })
        abstract_element = root.find('.//abstract')
        if abstract_element is not None:
            abstract_text = ' '.join([p.text for p in abstract_element.findall('.//p') if p.text])
            if abstract_text:
                structured_text.append({
                    'section': 'Abstract',
                    'text': abstract_text.strip()
                })
        body = root.find('.//body')
        if body is not None:
            # Find all <sec> tags directly under the body
            for section in body.findall('sec'):
                sec_title_element = section.find('title')
                sec_title = sec_title_element.text.strip() if sec_title_element is not None and sec_title_element.text else 'Introduction' # Default title
                
                # Get all paragraph text within this section
                paragraphs = [p.text for p in section.findall('.//p') if p.text]
                section_text = ' '.join(paragraphs).strip()

                if section_text:
                    structured_text.append({
                        'section': sec_title,
                        'text': section_text
                    })
        return structured_text
    except ET.ParseError:
        print(f"Warning: Could not parse XML file: {file_path}. Skipping.")
        return None
