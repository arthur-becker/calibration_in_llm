echo "PIPELINE 2 HAS STARTED" >> pipeline2.log

# llama-2-7b-f16-wikitext
python main.py llama-2-7b-f16-wiki-train llama-2-7b-f16-wiki-test llama-2-7b-f16-wikitext --calibration_steps 10
echo "llama-2-7b-f16-wikitext" >> pipeline2.log

# llama-2-7b-q4_0-wikitext
python main.py llama-2-7b-q4_0-wiki-train llama-2-7b-q4_0-wiki-test llama-2-7b-q4_0-wikitext --calibration_steps 10
echo "llama-2-7b-q4_0-wikitext" >> pipeline2.log

# llama-2-7b-q4_1-wikitext
python main.py llama-2-7b-q4_1-wiki-train llama-2-7b-q4_1-wiki-test llama-2-7b-q4_1-wikitext --calibration_steps 10
echo "llama-2-7b-q4_1-wikitext" >> pipeline2.log

# llama-2-7b-q8_0-wikitext
python main.py llama-2-7b-q8_0-wiki-train llama-2-7b-q8_0-wiki-test llama-2-7b-q8_0-wikitext --calibration_steps 10
echo "llama-2-7b-q8_0-wikitext" >> pipeline2.log

# llama-2-13b-f16-wikitext
python main.py llama-2-13b-f16-wiki-train llama-2-13b-f16-wiki-test llama-2-13b-f16-wikitext --calibration_steps 10
echo "llama-2-13b-f16-wikitext" >> pipeline2.log

# llama-2-13b-q4_0-wikitext
python main.py llama-2-13b-q4_0-wiki-train llama-2-13b-q4_0-wiki-test llama-2-13b-q4_0-wikitext --calibration_steps 10
echo "llama-2-13b-q4_0-wikitext" >> pipeline2.log

# llama-2-13b-q4_1-wikitext
python main.py llama-2-13b-q4_1-wiki-train llama-2-13b-q4_1-wiki-test llama-2-13b-q4_1-wikitext --calibration_steps 10
echo "llama-2-13b-q4_1-wikitext" >> pipeline2.log

# llama-2-13b-q8_0-wikitext
python main.py llama-2-13b-q8_0-wiki-train llama-2-13b-q8_0-wiki-test llama-2-13b-q8_0-wikitext --calibration_steps 10
echo "llama-2-13b-q8_0-wikitext" >> pipeline2.log

echo "PIPELINE 2 HAS FINISHED" >> pipeline2.log
