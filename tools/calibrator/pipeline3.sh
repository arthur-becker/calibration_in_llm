echo "PIPELINE 3 HAS STARTED" > pipeline3.log

### ARXIV
python main.py llama-2-13b-f16-calibration_arxiv-train llama-2-13b-f16-calibration_arxiv-test llama-2-13b-f16-calibration_arxiv
echo "llama-2-7b-f16-wikitext" >> pipeline2.log

python main.py llama-2-13b-q4_0-calibration_arxiv-train llama-2-13b-q4_0-calibration_arxiv-test llama-2-13b-q4_0-calibration_arxiv
echo "llama-2-7b-q4_0-wikitext" >> pipeline2.log

python main.py llama-2-13b-q8_0-calibration_arxiv-train llama-2-13b-q8_0-calibration_arxiv-test llama-2-13b-q4_1-calibration_arxiv
echo "llama-2-7b-q4_1-wikitext" >> pipeline2.log

python main.py llama-2-7b-f16-calibration_arxiv-train llama-2-7b-f16-calibration_arxiv-test llama-2-7b-f16-calibration_arxiv
echo "llama-2-7b-q8_0-wikitext" >> pipeline2.log

python main.py llama-2-7b-q4_0-calibration_arxiv-train llama-2-7b-q4_0-calibration_arxiv-test llama-2-7b-q4_0-calibration_arxiv
echo "llama-2-13b-f16-wikitext" >> pipeline2.log

python main.py llama-2-7b-q4_1-calibration_arxiv-train llama-2-7b-q4_1-calibration_arxiv-test llama-2-7b-q4_1-calibration_arxiv
echo "llama-2-13b-q4_0-wikitext" >> pipeline2.log

# Code Mistral
python main.py code-mistral-7b-q4_0-calibration_wikitext_top_40-train code-mistral-7b-q4_0-calibration_wikitext_top_40-test code-mistral-7b-q4_0-calibration_wikitext_top_40
echo "code-mistral-7b-q4_0-calibration_wikitext_top_40" >> pipeline2.log

python main.py code-mistral-7b-q4_0-calibration_wikitext_top_40-train code-mistral-7b-q4_0-gsm8k-test code-mistral-7b-q4_0-calibration_wikitext_gsm8k_CROSSVALID
echo "code-mistral-7b-q4_0-calibration_wikitext_gsm8k_CROSSVALID" >> pipeline2.log

# Cross validation: Wikitext -> Arxiv
python main.py llama-2-7b-f16-calibration_wikitext-train llama-2-7b-f16-calibration_arxiv-test llama-2-7b-f16-calibration_wikitext-arxiv-CROSSVALID
echo "llama-2-7b-f16-calibration_wikitext-arxiv-CROSSVALID" >> pipeline2.log

python main.py llama-2-7b-q4_0-calibration_wikitext-train llama-2-7b-q4_0-calibration_arxiv-test llama-2-7b-q4_0-calibration_wikitext-arxiv-CROSSVALID
echo "llama-2-7b-q4_0-calibration_wikitext-arxiv-CROSSVALID" >> pipeline2.log

python main.py llama-2-7b-q8_0calibration_wikitext-train llama-2-7b-q8_0-calibration_arxiv-test llama-2-7b-q8_0-calibration_wikitext-arxiv-CROSSVALID
echo "llama-2-7b-q8_0-calibration_wikitext-arxiv-CROSSVALID" >> pipeline2.log

# FINISHED
echo "PIPELINE 3 HAS FINISHED" >> pipeline3.log