import re
import sys
from jiwer import wer, cer, process_words

import unicodedata

# bangla digit to arabic digit mapping to unify numeral styles before comparing.
bangla_to_arabic_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# characters to strip entirely, including bangla dari and standard punctuation.
punctuation_pattern = re.compile(r"[।,.!?;:\"'()\[\]{}—–\-]")
bracket_tag_pattern = re.compile(r"\[[^\]]*\]")


def normalize_transcript(text_content: str) -> str:
    # apply identical cleanup to reference and hypothesis before scoring.
    text_content = bracket_tag_pattern.sub(" ", text_content)
    text_content = text_content.translate(bangla_to_arabic_digits)
    text_content = punctuation_pattern.sub(" ", text_content)

    text_content = unicodedata.normalize("NFC", text_content)
    text_content = text_content.replace("\u200c", "").replace("\u200d", "")
    text_content = re.sub(r"\s+", " ", text_content).strip()
    return text_content


def load_text_file(file_path: str) -> str:
    # reads the content of a text file safely.
    with open(file_path, "r", encoding="utf-8") as target_file:
        return target_file.read()


def main():
    # executes the core evaluation logic.
    if len(sys.argv) != 3:
        print("usage: python compute_wer_cer.py <reference.txt> <hypothesis.txt>")
        sys.exit(1)

    reference_file_path = sys.argv[1]
    hypothesis_file_path = sys.argv[2]

    raw_reference_text = load_text_file(reference_file_path)
    raw_hypothesis_text = load_text_file(hypothesis_file_path)

    normalized_reference = normalize_transcript(raw_reference_text)
    normalized_hypothesis = normalize_transcript(raw_hypothesis_text)

    if not normalized_reference or not normalized_hypothesis:
        print("[!] one of the normalized transcripts is empty. please check your input files.")
        sys.exit(1)

    # calculate the error rates under the hood using the jiwer library.
    word_error_rate_score = wer(normalized_reference, normalized_hypothesis)
    character_error_rate_score = cer(normalized_reference, normalized_hypothesis)
    detailed_word_statistics = process_words(normalized_reference, normalized_hypothesis)

    print("=" * 60)
    print(f"reference file:  {reference_file_path}")
    print(f"hypothesis file: {hypothesis_file_path}")
    print("=" * 60)
    print(f"wer (word error rate):      {word_error_rate_score:.4f}  ({word_error_rate_score * 100:.2f}%)")
    print(f"cer (character error rate): {character_error_rate_score:.4f}  ({character_error_rate_score * 100:.2f}%)")
    print("-" * 60)
    print(f"substitutions (words): {detailed_word_statistics.substitutions}")
    print(f"deletions (words):     {detailed_word_statistics.deletions}")
    print(f"insertions (words):    {detailed_word_statistics.insertions}")
    print(f"hits (words):          {detailed_word_statistics.hits}")
    print(f"reference word count:  {len(normalized_reference.split())}")
    print("=" * 60)
    print("\nnote: these metrics reflect transcription accuracy after stripping punctuation and tags.")


if __name__ == "__main__":
    main()