import http.server
import socketserver
from http import HTTPStatus
PORT = 7777

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'Hello world')


httpd = socketserver.TCPServer(('', PORT), Handler)
print("Serving a server at: ", PORT)
httpd.serve_forever()
