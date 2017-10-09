def top_words_html(sklearn_topic_model, feature_names, n_top_words, html_file):
    for topic_idx, topic in enumerate(sklearn_topic_model.components_):
        html_file.write("<h3> Topic "+str(topic_idx+1)+" </h3>\n")
	html_file.write("<p> ")
	for i in topic.argsort()[:-n_top_words - 1:-1]:
		html_file.write(feature_names[i].encode("UTF-8"))
		html_file.write(' ')	
	html_file.write("</p>\n")

def init_html_file(file_name, city_name):
	if (len(file_name) < 5) or (file_name[-5:] != ".html"):
		file_name = file_name + ".html"
	html_file = open(file_name, 'w')
	html_file.write("<!doctype html>\n")
	html_file.write("<html>\n")
	html_file.write("<body>\n")
	html_file.write("<h1> "+city_name+" </h1>\n")
	
	return html_file

def add_html_h2(html_file, h2_text):
	html_file.write("<h2> "+h2_text+" </h2>\n")


def end_html_file(html_file):
	html_file.write("</body>\n")
	html_file.write("</html>\n")
	
	html_file.close()

def add_html_image(image_name, html_file, add_dot_png=False, width_px=640, height_px=480):
	if add_dot_png and ((len(image_name) < 4) or (image_name[-4:] != ".png")):
		image_name = image_name + ".png"
	html_file.write("<p> ")
	html_file.write("<img src=\""+image_name+"\" alt=\""+image_name+"\" style=\"width:"+str(width_px)+"px;height:"+str(height_px)+"px;\"> ")
	html_file.write("</p>\n")

def add_html_link(html_file, link_url, link_text, add_dot_html=False):
	if add_dot_html and ((len(link_url) < 5) or (link_url[-5:] != ".html")):
		link_url = link_url + ".html"
	html_file.write("<p> ")
	html_file.write("<a href=\""+link_url+"\">"+link_text+"</a> ")
	html_file.write("</p>\n")

	
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud_image(text_data, output_file_name):
	if (len(output_file_name) < 4) or (output_file_name[-4:] != ".png"):
		output_file_name = output_file_name + ".png"
	wordcloud = WordCloud(max_font_size=40).generate(text_data)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.savefig(output_file_name)
	plt.close()

