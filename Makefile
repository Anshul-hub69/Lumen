.PHONY: install data train eval test serve all
install: ; pip install -r requirements.txt
data:    ; python -m lumen.cli generate
train:   ; python -m lumen.cli train
eval:    ; python -m lumen.cli evaluate
test:    ; python -m unittest discover -s tests -t . -v
serve:   ; python -m lumen.cli serve --port 8000
all:     ; python -m lumen.cli all
