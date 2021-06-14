import json, csv
import os

from webpie import WPHandler, WPApp, HTTPServer, WPStaticHandler, Response

class Monitor(object):
    
    def __init__(self, fn, title="Monitor", plots = [], metadata={}):
        #
        # plot_list: [ plot_desc, ... ]
        # plot_desc: [ series, ... ]
        # series:
        # {
        #   label: "label",
        #   secondary_axis: <bool>,                 default: false
        #   line_width:     <float> or 0.0,         default: 1.0
        #   color:     <string>                default: auto
        #   marker_style:   <string> or null,       default: null
        # }
        #
        self.FileName = fn
        self.Labels = set()
        self.Data = []          # [(t, data_dict), ...]
        self.SaveInterval = 1
        self.NextSave = self.SaveInterval
        self.Server = None
        self.PlotDesc = plots
        self.Title = title
        self.Meta = metadata
        
    def reset(self):
        self.Labels = set()
        self.Data = []
        self.NextSave = self.SaveInterval        
        
    def start_server(self, port):
        app = App(self, static_location="static", enable_static=True)    
        self.Server = HTTPServer(port, app, logging=False)
        self.Server.start()
        return self.Server
        
    def add(self, t, data=None, **data_args):
        if data is None:    data = data_args
        self.Data.append((t, data.copy()))
        for k in data.keys():
            self.Labels.add(k)
        self.NextSave -= 1
        if self.NextSave <= 0:
            self.save()
            self.NextSave = self.SaveInterval

    def data_as_columns(self):
        labels = list(self.Labels)
        n = len(self.Data)
        prescale = 1.0 if n < 10000 else 0.1
        scaler = 0.0
        columns = {c:[] for c in labels}
        columns['t'] = []
        for i, (t, row) in enumerate(self.Data):
            scaler += prescale
            if scaler >= 1.0 or i == n-1:
                scaler -= 1.0
                columns['t'].append(t)
                for l in labels:
                    columns[l].append(row.get(l))
        return columns
        
    def data_as_rows(self):
        out = []
        for t, dct in self.Data:
            d = dct.copy()
            d["t"] = t
            out.append(d)
        return out
            
    def save(self):
        return
        labels, rows = self.data_as_columns()
        with open(self.FileName, "w") as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            for row in rows:
                writer.writerow(row)

class Handler(WPHandler):
    
    def __init__(self, request, app):
        WPHandler.__init__(self, request, app)
        #print("Monitor: module home:", os.path.dirname(__file__))
        self.static = WPStaticHandler(request, app, root=os.path.dirname(__file__)+"/static")
    
    def data(self, request, relpath, **args):
        columns = self.App.Monitor.data_as_columns()
        rows = self.App.Monitor.data_as_rows()
        out = {
            "labels":       sorted(list(columns.keys())),
            "columns":      columns,
            "rows":         rows,
            "plots":        self.App.Monitor.PlotDesc,
            "title":        self.App.Monitor.Title,
            "metadata":     self.App.Monitor.Meta
        }
        response = Response(json.dumps(out), headers={
            "Content-Type": "text/json",
            "Access-Control-Allow-Origin": "*"
        })
        return response
        
class App(WPApp):
    
    def __init__(self, mon, **args):
        WPApp.__init__(self, Handler, **args)
        self.Monitor = mon
        
if __name__ == "__main__":
    import time, random
    
    m = Monitor("/dev/null", 
        title = "Plots demo",
        plots = [
            [
                {
                    "label":    "x",
                    "line_width": 0.5
                },
                {
                    "label":    "y",
                    "line_width":   1.0,
                    "marker_style": "Circle"
                }
            ],
            [
                {
                    "label":    "x+y",
                    "color":    "gray"
                },
                {
                    "label":    "x-y",
                    "color":    "green"
                }                
            ]
        ])
    m.start_server(8888)
    
    while True:
        m.reset()
        x = 0.0
        for t in range(100):
            y = random.random() - 0.48
            x += y
            m.add(t, data={
                "x":x, "y":y, "x+y":x+y, "x-y":x-y
            })
        time.sleep(5)
        
        