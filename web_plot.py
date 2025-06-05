from flask import Flask, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from sql_plot_mib import SQLPlotter
import tempfile

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '16'))
ALLOWED_EXTENSIONS = {'csv'}

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024  # Upload limit from env var

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'files[]' not in request.files:
                return redirect(request.url)
            
            files = request.files.getlist('files[]')
            filenames = []
            
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    filenames.append(filepath)
            
            if filenames:
                # Create temporary database path
                db_path = os.path.join(tempfile.gettempdir(), 'uploaded_data.db')
                
                # Process uploaded files
                plotter = SQLPlotter(data_dir=os.path.dirname(filenames[0]), db_path=db_path)
                
                # Generate HTML plot
                output_path = os.path.join(tempfile.gettempdir(), 'plot_output.html')
                plotter.create_interactive_plot(output_path=output_path, show_plot=False)
                
                # Read generated HTML
                with open(output_path, 'r') as f:
                    plot_html = f.read()
                
                return render_template('plot.html', plot_html=plot_html)
        
        return render_template('upload.html')

    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'plot-app'}, 200

    return app

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app = create_app()
    app.run(port=5000)
