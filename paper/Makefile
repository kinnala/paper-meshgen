default: image_1.png
	-rm meshgen.{aux,bbl,blg}
	pdflatex meshgen.tex
	bibtex meshgen
	pdflatex meshgen.tex
	pdflatex meshgen.tex
	zip arxiv_export.zip meshgen.tex meshgen.bbl siamplain.bst `rg -g '!Makefile' image_ | sed -e 's/.*\(image_[0-9]*.png\).*/\1/' | sort | uniq | tr '\n' ' '`

image_1.png:
	unzip -o example_meshes.zip

clean:
	-rm meshgen.{aux,bbl,blg}
	-rm *.png
