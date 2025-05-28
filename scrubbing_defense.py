# scrubbing_defense.py

import spacy
import json

nlp = spacy.load("en_core_web_sm")

def scrub_text(text):
    doc = nlp(text)
    scrubbed = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "ORG"]:
            scrubbed = scrubbed[:ent.start_char] + "<NAME>" + scrubbed[ent.end_char:]
        elif ent.label_ == "DATE":
            scrubbed = scrubbed[:ent.start_char] + "<DATE>" + scrubbed[ent.end_char:]
        elif ent.label_ == "GPE":
            scrubbed = scrubbed[:ent.start_char] + "<LOCATION>" + scrubbed[ent.end_char:]
        elif ent.label_ == "EMAIL":
            scrubbed = scrubbed[:ent.start_char] + "<EMAIL>" + scrubbed[ent.end_char:]
    return scrubbed

def scrub_enron_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            text = data.get("text", "")
            data["text"] = scrub_text(text)
            outfile.write(json.dumps(data) + "\n")
    print(f"✅ Scrubbed Enron saved to: {output_file}")
