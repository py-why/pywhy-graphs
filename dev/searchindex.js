Search.setIndex({"docnames": ["api", "auto_examples/intro_causal_graphs", "generated/pywhy_graphs.ADMG", "generated/pywhy_graphs.CPDAG", "generated/pywhy_graphs.PAG", "generated/pywhy_graphs.algorithms.acyclification", "generated/pywhy_graphs.algorithms.discriminating_path", "generated/pywhy_graphs.algorithms.is_valid_mec_graph", "generated/pywhy_graphs.algorithms.pds", "generated/pywhy_graphs.algorithms.pds_path", "generated/pywhy_graphs.algorithms.possible_ancestors", "generated/pywhy_graphs.algorithms.possible_descendants", "generated/pywhy_graphs.algorithms.uncovered_pd_path", "generated/pywhy_graphs.array.clearn_arr_to_graph", "generated/pywhy_graphs.array.graph_to_arr", "index", "installation", "use", "whats_new", "whats_new/_contributors", "whats_new/v0.1"], "filenames": ["api.rst", "auto_examples/intro_causal_graphs.rst", "generated/pywhy_graphs.ADMG.rst", "generated/pywhy_graphs.CPDAG.rst", "generated/pywhy_graphs.PAG.rst", "generated/pywhy_graphs.algorithms.acyclification.rst", "generated/pywhy_graphs.algorithms.discriminating_path.rst", "generated/pywhy_graphs.algorithms.is_valid_mec_graph.rst", "generated/pywhy_graphs.algorithms.pds.rst", "generated/pywhy_graphs.algorithms.pds_path.rst", "generated/pywhy_graphs.algorithms.possible_ancestors.rst", "generated/pywhy_graphs.algorithms.possible_descendants.rst", "generated/pywhy_graphs.algorithms.uncovered_pd_path.rst", "generated/pywhy_graphs.array.clearn_arr_to_graph.rst", "generated/pywhy_graphs.array.graph_to_arr.rst", "index.rst", "installation.md", "use.rst", "whats_new.rst", "whats_new/_contributors.rst", "whats_new/v0.1.rst"], "titles": ["API", "An introduction to causal graphs and how to use them", "pywhy_graphs.ADMG", "pywhy_graphs.CPDAG", "pywhy_graphs.PAG", "pywhy_graphs.algorithms.acyclification", "pywhy_graphs.algorithms.discriminating_path", "pywhy_graphs.algorithms.is_valid_mec_graph", "pywhy_graphs.algorithms.pds", "pywhy_graphs.algorithms.pds_path", "pywhy_graphs.algorithms.possible_ancestors", "pywhy_graphs.algorithms.possible_descendants", "pywhy_graphs.algorithms.uncovered_pd_path", "pywhy_graphs.array.clearn_arr_to_graph", "pywhy_graphs.array.graph_to_arr", "<strong>pywhy-graphs</strong>", "Installation", "Examples using pywhy-graphs", "Release History", "&lt;no title&gt;", "What\u2019s new?"], "terms": {"pywhy_graph": [0, 1, 20], "thi": [0, 1, 2, 3, 4, 5, 9, 14, 15, 18], "i": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16], "applic": 0, "program": 0, "interfac": 0, "refer": [0, 2, 3, 4, 5, 8], "camelcas": 0, "name": [0, 2, 3, 4, 5, 13], "function": [0, 1, 4], "underscore_cas": 0, "pywhi": [0, 16, 18, 20], "group": 0, "themat": 0, "analysi": 0, "stage": 0, "These": [0, 1, 8], "ar": [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 14, 18], "structur": [0, 2, 3, 4, 14, 15], "model": [0, 3, 4], "scm": [0, 1, 3, 4], "variou": [0, 1, 20], "encount": 0, "literatur": [0, 1], "tradit": [0, 1, 2], "oper": [0, 5], "over": [0, 2, 3, 4], "onli": [0, 1, 2, 3, 4], "one": [0, 1, 2, 3, 4, 6, 7, 12, 13, 14], "type": [0, 2, 3, 4, 13], "edg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14], "gener": [0, 1, 2, 3, 4, 17], "consist": 0, "more": [0, 1, 2, 3, 4, 14, 15], "than": [0, 14], "common": [0, 1, 12], "varieti": 0, "differ": [0, 1, 2, 3, 4], "learn": [0, 1, 3, 4, 5, 8, 13, 14, 15, 17, 20], "implement": [0, 20], "infer": [0, 1, 8, 15], "procedur": [0, 5], "encod": [0, 1, 13, 14], "object": [0, 1, 2, 3, 4, 13], "submodul": 0, "convert": [0, 5, 13, 14, 20], "those": [0, 2, 3, 4], "data": [0, 2, 3, 4, 15], "correspond": [0, 2, 3, 4, 14], "click": 1, "here": [1, 2, 3, 4, 17, 20], "download": [1, 17], "full": 1, "exampl": [1, 14, 15], "code": [1, 17], "graphic": 1, "attach": 1, "notion": 1, "each": [1, 2, 3, 4, 7], "miss": 1, "we": [1, 12, 14, 15, 17, 20], "review": 1, "fundament": 1, "from": [1, 2, 3, 4, 6, 8, 12, 16], "import": 1, "networkx": [1, 2, 3, 4, 15], "nx": [1, 2, 3, 4, 5, 13, 14], "numpi": [1, 14], "np": 1, "panda": 1, "pd": [1, 9, 12, 20], "dowhi": 1, "gcm": 1, "util": 1, "set_random_se": 1, "scipi": 1, "stat": 1, "viz": 1, "draw": 1, "1": [1, 2, 3, 4, 5, 8, 9, 12, 18], "mathemat": 1, "defin": [1, 8, 9, 12], "4": [1, 2, 3, 4, 9], "tupl": [1, 2, 3, 4, 6, 12, 14], "v": [1, 2, 3, 4, 6, 8, 12, 14], "u": [1, 2, 3, 4, 6, 8, 12, 14], "f": 1, "p": [1, 6], "where": [1, 2, 3, 4, 6, 7, 12], "set": [1, 2, 3, 4, 6, 8, 9, 10, 11, 12], "endogen": 1, "observ": 1, "variabl": [1, 8], "exogen": 1, "unobserv": [1, 3, 4], "everi": [1, 6, 8, 12], "distribut": 1, "all": [1, 2, 3, 4, 5, 8, 15, 17, 18], "taken": [1, 2, 3, 4], "togeth": 1, "four": [1, 2, 3, 4], "mechan": 1, "problem": 1, "almost": 1, "alwai": 1, "howev": [1, 12], "induc": [1, 2, 3, 4], "which": [1, 2, 3, 4, 6, 8, 9, 12, 13, 14, 16], "ha": [1, 2, 3, 4, 6, 8, 20], "node": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "argument": [1, 2, 3, 4], "If": [1, 2, 3, 4, 8, 12], "parent": [1, 2, 3, 4, 6], "ani": [1, 2, 3, 4, 5], "can": [1, 2, 3, 4, 14], "repres": [1, 2, 3, 4, 14, 15], "bidirect": [1, 2, 3, 4, 5, 12], "The": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "latent": [1, 2, 3, 4, 8], "confound": [1, 2, 3, 4], "between": [1, 2, 3, 4, 6, 7, 8, 13, 14, 20], "two": [1, 2, 3, 4, 12], "even": 1, "though": [1, 2, 3, 4], "typic": [1, 12], "unknown": 1, "still": 1, "ground": 1, "truth": 1, "evalu": 1, "algorithm": [1, 3, 4, 15, 20], "build": [1, 15], "our": [1, 15], "intuit": 1, "understand": 1, "context": 1, "random": 1, "seed": 1, "make": [1, 5], "reproduc": [1, 2, 3, 4], "12345": 1, "rng": 1, "randomst": 1, "class": [1, 2, 3, 4], "mycustommodel": 1, "predictionmodel": 1, "def": 1, "__init__": 1, "self": 1, "coeffici": 1, "fit": 1, "x": [1, 2, 3, 4, 8], "y": [1, 2, 3, 8], "noth": 1, "sinc": [1, 20], "know": 1, "pass": [1, 3, 4, 12], "predict": 1, "return": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], "clone": [1, 16], "don": [1, 2, 3, 4], "t": [1, 2, 3, 4], "realli": 1, "need": [1, 12, 17], "actual": [1, 2, 3, 4], "1234": 1, "construct": 1, "result": [1, 2, 3, 4], "xy": 1, "z": [1, 8], "w": 1, "_________": 1, "g": [1, 2, 3, 4, 5, 7, 10, 11, 14], "digraph": [1, 2, 3, 4], "dot_graph": 1, "render": 1, "outfil": 1, "png": 1, "view": [1, 2, 3, 4], "true": [1, 2, 3, 4, 5], "causal_model": 1, "probabilisticcausalmodel": 1, "set_causal_mechan": 1, "scipydistribut": 1, "binom": 1, "0": [1, 2, 3, 4, 18], "5": [1, 2, 3, 4], "n": [1, 2, 3, 4], "9": [1, 20], "additivenoisemodel": 1, "prediction_model": 1, "noise_model": 1, "8": [1, 16], "would": [1, 14], "paramet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "do": [1, 2, 3, 4, 7], "anyth": 1, "method": [1, 2, 3, 4], "ensur": 1, "fcm": 1, "correct": [1, 2, 3, 4], "local": 1, "hash": 1, "e": [1, 2, 3, 4, 5, 8, 16], "get": [1, 2, 3, 4], "inconsist": 1, "error": [1, 7], "modifi": 1, "afterward": 1, "without": [1, 2, 3, 4, 5], "updat": [1, 2, 3, 4], "have": [1, 2, 3, 4, 7, 14], "empti": [1, 2, 3, 4], "small": 1, "workaround": 1, "pre": 1, "datafram": 1, "column": [1, 13, 14], "sampl": 1, "draw_sampl": 1, "num_sampl": 1, "500": 1, "print": 1, "head": 1, "seri": 1, "col": 1, "uniqu": 1, "note": [1, 2, 3, 4, 5, 8, 9, 12, 14, 18], "show": 1, "collid": [1, 6, 8], "collis": 1, "arrow": [1, 12], "00": 1, "": [1, 2, 3, 4, 8, 15, 18], "100": 1, "1608": 1, "12it": 1, "3": [1, 2, 3, 4, 6, 9, 16], "2": [1, 2, 3, 4, 14], "row": [1, 13, 14], "dtype": [1, 13, 14], "graphviz": 1, "0x7fe9974485b0": 1, "markovian": [1, 3, 4], "suffici": [1, 3, 4], "assumpt": [1, 3, 4], "One": [1, 3, 4, 14], "queri": [1, 2, 3, 4], "list": [1, 2, 3, 4, 6, 12, 13, 14, 20], "predecessor": [1, 2, 3, 4], "Or": 1, "children": [1, 2, 3, 4], "successor": [1, 2, 3, 4], "explor": [1, 6], "d": [1, 2, 3, 4, 8, 9], "separ": [1, 8, 9], "statement": 1, "markov": [1, 3, 4], "condit": 1, "impli": [1, 3], "independ": [1, 2, 3, 4], "For": [1, 2, 3, 4, 14, 15], "becaus": 1, "d_separ": 1, "open": 1, "up": 1, "path": [1, 2, 3, 4, 6, 8, 9, 12], "given": [1, 2, 3, 4], "fals": [1, 2, 3, 4], "semi": 1, "possibli": [1, 8, 9], "depict": 1, "just": [1, 2, 3, 4, 16], "set_nodes_as_latent_confound": 1, "now": 1, "anymor": 1, "form": [1, 2, 3, 4, 7, 8], "cluster": 1, "what": [1, 2, 3, 4, 5, 18], "compon": [1, 2, 4, 5, 9], "c": [1, 2, 3, 4, 6, 12], "short": 1, "c_compon": [1, 2, 4], "look": [1, 17], "m": [1, 5], "similarli": [1, 2, 3, 4], "m_separ": 1, "sai": 1, "add": [1, 2, 3, 4, 14, 16], "thei": [1, 2, 3, 4, 14], "longer": 1, "add_edg": [1, 2, 3, 4], "bidirected_edge_nam": [1, 2, 4], "equival": [1, 2, 3, 4], "besid": [1, 12], "relationship": [1, 2, 3, 14], "other": [1, 2, 3, 4, 7], "same": [1, 2, 3, 4], "commonli": 1, "constraint": [1, 3, 4, 5], "base": [1, 3, 4, 5], "seek": 1, "reconstruct": 1, "part": 1, "In": [1, 4, 5, 12, 20], "next": 1, "section": 1, "briefli": 1, "overview": [1, 17], "usual": [1, 12], "so": [1, 2, 3, 4, 14], "dodiscov": 1, "http": [1, 2, 3, 4, 5, 8, 16], "github": [1, 16], "com": [1, 16], "py": [1, 16], "why": [1, 16], "_": 1, "detail": 1, "discoveri": [1, 5], "pleas": 1, "see": [1, 2, 3, 4, 14, 15], "repo": 1, "stem": 1, "relev": [1, 2, 3, 4, 15], "assum": 1, "uncertain": [1, 8], "orient": [1, 3, 4, 6, 12], "via": [1, 2, 3, 4, 16], "undirect": [1, 2, 3, 4, 14], "ll": 1, "earlier": 1, "learnt": 1, "variant": 1, "pc": [1, 3, 4], "let": 1, "add_edges_from": [1, 2, 3, 4], "undirected_edge_nam": [1, 2, 3, 4], "unshield": [1, 12], "present": [1, 2, 5, 14], "origin": [1, 2, 3, 4], "orient_uncertain_edg": [1, 3, 4], "allow": [1, 3, 4, 8, 14], "complex": [1, 13, 14], "compar": [1, 9], "circl": [1, 4, 12], "endpoint": [1, 4, 6, 12, 13, 14], "A": [1, 2, 3, 4, 5, 6, 7, 12], "o": [1, 4, 12], "either": [1, 2, 3, 4, 5], "tail": [1, 14], "arrowhead": [1, 3, 4], "select": [1, 2, 8, 14], "bia": [1, 2, 14], "possibl": [1, 4, 10, 11, 14], "presenc": [1, 2, 5, 14], "fci": [1, 12], "footcit": 1, "zhang2008": 1, "circle_edge_nam": [1, 4], "judea": 1, "pearl": 1, "reason": 1, "cambridg": 1, "univers": 1, "press": [1, 5], "usa": 1, "2nd": 1, "edit": 1, "2009": 1, "isbn": 1, "052189560x": 1, "peter": [1, 5], "spirt": 1, "clark": 1, "glymour": 1, "richard": 1, "schein": 1, "causat": 1, "search": [1, 8, 9], "volum": [1, 5], "81": 1, "mit": 1, "01": 1, "1993": 1, "978": 1, "4612": 1, "7650": 1, "doi": [1, 8], "10": [1, 8, 20], "1007": 1, "2748": 1, "total": [1, 2, 3, 4], "run": [1, 16], "time": [1, 2, 3, 4], "script": 1, "minut": 1, "949": 1, "second": [1, 2, 3, 4], "python": [1, 2, 3, 4, 15, 16, 17], "sourc": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17], "intro_causal_graph": 1, "jupyt": [1, 17], "notebook": [1, 17], "ipynb": 1, "galleri": [1, 17], "sphinx": [1, 17], "incoming_directed_edg": [2, 3, 4], "none": [2, 3, 4, 6, 8, 9, 12, 14], "incoming_bidirected_edg": [2, 4], "incoming_undirected_edg": [2, 3, 4], "directed_edge_nam": [2, 3, 4], "str": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "direct": [2, 3, 4, 5, 7, 8, 12, 14, 15], "attr": [2, 3, 4], "acycl": [2, 3, 8, 15, 20], "mix": [2, 3, 4, 14, 15], "graph": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20], "causal": [2, 3, 4, 5, 13, 14, 15, 17, 20], "constitut": 2, "relat": 2, "dag": [2, 3, 4, 13, 15], "did": 2, "while": [2, 3, 4], "input": [2, 3, 4], "option": [2, 3, 4, 6, 7, 8, 9, 12, 13, 14], "default": [2, 3, 4, 6, 7, 8, 9, 12, 14], "initi": [2, 3, 4], "accept": [2, 3, 4], "By": [2, 3, 4, 6, 8, 12], "keyword": [2, 3, 4], "attribut": [2, 3, 4], "kei": [2, 3, 4], "valu": [2, 3, 4], "pair": [2, 3, 4, 7], "mixededgegraph": [2, 3, 4, 5, 13, 14, 15], "underneath": 2, "hood": 2, "store": [2, 13], "non": [2, 3, 4, 6, 7], "an": [2, 3, 4, 5, 7, 8, 12, 14, 17, 20], "stand": 2, "normal": 2, "indic": [2, 14], "intern": [2, 3, 4, 7], "bidirected_edg": [2, 4], "edgeview": [2, 3, 4], "directed_edg": [2, 3, 4], "edge_typ": [2, 3, 4], "string": [2, 3, 4], "identifi": [2, 3, 4], "undirected_edg": [2, 3, 4], "add_edge_typ": [2, 3, 4], "add_edge_types_from": [2, 3, 4], "add_nod": [2, 3, 4], "add_nodes_from": [2, 3, 4], "clear": [2, 3, 4], "clear_edg": [2, 3, 4], "is_mix": [2, 3, 4], "neighbor": [2, 3, 4], "remove_edge_typ": [2, 3, 4], "remove_nod": [2, 3, 4], "remove_nodes_from": [2, 3, 4], "__contains__": [2, 3, 4], "otherwis": [2, 3, 4], "path_graph": [2, 3, 4], "multigraph": [2, 3, 4], "multidigraph": [2, 3, 4], "etc": [2, 3, 4], "__getitem__": [2, 3, 4], "dict": [2, 3, 4], "adj_dict": [2, 3, 4], "dictionari": [2, 3, 4], "adjac": [2, 3, 4, 6, 8, 9, 12], "connect": [2, 3, 4, 5], "adj": [2, 3, 4], "similar": [2, 3, 4], "iter": [2, 3, 4], "atlasview": [2, 3, 4], "__iter__": [2, 3, 4], "niter": [2, 3, 4], "__len__": [2, 3, 4], "number": [2, 3, 4], "len": [2, 3, 4], "nnode": [2, 3, 4], "int": [2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14], "number_of_nod": [2, 3, 4], "ident": [2, 3, 4], "order": [2, 3, 4, 13, 14], "u_of_edg": [2, 3, 4], "v_of_edg": [2, 3, 4], "automat": [2, 3, 4], "ad": [2, 3, 4, 14], "alreadi": [2, 3, 4], "specifi": [2, 3, 4, 14], "directli": [2, 3, 4], "access": [2, 3, 4], "below": [2, 3, 4], "u_for_edg": [2, 3, 4], "v_for_edg": [2, 3, 4], "must": [2, 3, 4, 12, 13], "hashabl": [2, 3, 4], "label": [2, 3, 4], "assign": [2, 3, 4], "collect": [2, 3, 4], "ebunch_to_add": [2, 3, 4], "contain": [2, 3, 4, 9], "singl": [2, 3, 4], "twice": [2, 3, 4], "effect": [2, 3, 4, 17], "when": [2, 3, 4, 18], "duplic": [2, 3, 4], "ebunch": [2, 3, 4], "take": [2, 3, 4, 5, 9], "preced": [2, 3, 4], "zip": [2, 3, 4, 17], "rang": [2, 3, 4], "associ": [2, 3, 4], "weight": [2, 3, 4], "wn2898": [2, 3, 4], "properti": [2, 3, 4], "hold": [2, 3, 4], "inform": [2, 3, 4, 14], "itself": [2, 3, 4], "document": [2, 3, 4], "behav": [2, 3, 4], "like": [2, 3, 4, 14], "idiom": [2, 3, 4], "includ": [2, 3, 4, 20], "follow": [2, 3, 4, 8], "loop": [2, 3, 4], "item": [2, 3, 4], "nbr": [2, 3, 4], "datadict": [2, 3, 4], "main": [2, 3, 4], "within": [2, 3, 4, 12], "adjacencyview": [2, 3, 4], "map": [2, 3, 4], "trivial": [2, 4], "themself": [2, 4], "comp": [2, 4], "union": [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], "float": [2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14], "consid": [2, 3, 4, 12], "clear_edge_typ": [2, 3, 4], "copi": [2, 3, 4, 5], "shallow": [2, 3, 4], "That": [2, 3, 4], "share": [2, 3, 4], "deepcopi": [2, 3, 4], "new": [2, 3, 4, 18], "as_view": [2, 3, 4], "bool": [2, 3, 4, 5, 6, 7, 12, 13, 14], "provid": [2, 3, 4], "read": [2, 3, 4], "to_direct": [2, 3, 4], "mai": [2, 3, 4], "handl": [2, 3, 4], "wai": [2, 3, 4], "There": [2, 3, 4, 12], "peopl": [2, 3, 4], "might": [2, 3, 4], "want": [2, 3, 4], "well": [2, 3, 4, 14, 15], "entir": [2, 3, 4], "chang": [2, 3, 4, 18, 19, 20], "affect": [2, 3, 4], "save": [2, 3, 4], "memori": [2, 3, 4], "could": [2, 3, 4], "caus": [2, 3, 4], "confus": [2, 3, 4], "you": [2, 3, 4, 15, 16, 17], "doe": [2, 3, 4, 12, 14], "level": [2, 3, 4], "creat": [2, 3, 4], "exactli": [2, 3, 4], "obtain": [2, 3, 4], "style": [2, 3, 4], "h": [2, 3, 4, 8], "__class__": [2, 3, 4], "fresh": [2, 3, 4], "enabl": [2, 3, 4], "instead": [2, 3, 4], "inspir": [2, 3, 4], "act": [2, 3, 4], "version": [2, 3, 4, 18], "requir": [2, 3, 4], "modul": [2, 3, 4, 15], "deep": [2, 3, 4], "doc": [2, 3, 4], "org": [2, 3, 4, 8, 16], "librari": [2, 3, 4, 18], "html": [2, 3, 4, 5], "degre": [2, 3, 4], "nbunch": [2, 3, 4], "degreeview": [2, 3, 4], "report": [2, 3, 4], "incid": [2, 3, 4], "numer": [2, 3, 4], "sum": [2, 3, 4], "deg_dict": [2, 3, 4], "multipl": [2, 3, 4], "request": [2, 3, 4, 15], "integ": [2, 3, 4], "ddict": [2, 3, 4], "quietli": [2, 3, 4], "ignor": [2, 3, 4, 7], "out": [2, 3, 4], "get_edge_data": [2, 3, 4], "except": [2, 3, 4], "doesn": [2, 3, 4], "exist": [2, 3, 4], "found": [2, 3, 4, 6], "edge_dict": [2, 3, 4], "warn": [2, 3, 4], "permit": [2, 3, 4], "But": [2, 3, 4], "safe": [2, 3, 4], "foo": [2, 3, 4], "7": [2, 3, 4], "b": [2, 3, 4], "get_graph": [2, 3, 4], "specif": [2, 3, 4, 15], "rais": [2, 3, 4, 7], "valueerror": [2, 3, 4], "_description_": [2, 3, 4], "graph_attr_dict_factori": [2, 3, 4], "alia": [2, 3, 4], "has_edg": [2, 3, 4], "keyerror": [2, 3, 4], "check": [2, 3, 4, 6, 7, 12], "edge_ind": [2, 3, 4], "data_dictionari": [2, 3, 4], "syntax": [2, 3, 4], "give": [2, 3, 4, 17], "has_nod": [2, 3, 4], "It": [2, 3, 4, 9], "readabl": [2, 3, 4], "simpler": [2, 3, 4], "is_direct": [2, 3, 4], "is_multigraph": [2, 3, 4], "appear": [2, 3, 4], "technic": [2, 3, 4], "user": [2, 3, 4], "control": [2, 3, 4], "nbunch_it": [2, 3, 4], "also": [2, 3, 4, 15], "membership": [2, 3, 4], "silent": [2, 3, 4], "networkxerror": [2, 3, 4], "sequenc": [2, 3, 4], "node_attr_dict_factori": [2, 3, 4], "node_dict_factori": [2, 3, 4], "number_of_edge_typ": [2, 3, 4], "number_of_edg": [2, 3, 4], "nedg": [2, 3, 4], "size": [2, 3, 4], "count": [2, 3, 4], "join": [2, 3, 4], "ancestor": [2, 3, 4, 10], "remove_edg": [2, 3, 4], "remov": [2, 3, 4], "remove_edges_from": [2, 3, 4], "k": [2, 3, 4], "Will": [2, 3, 4], "fail": [2, 3, 4], "6": [2, 3, 4, 20], "sub_bidirected_graph": [2, 4], "sub": [2, 3, 4, 5], "sub_directed_graph": [2, 3, 4], "sub_undirected_graph": [2, 3, 4], "subgraph": [2, 3, 4, 8], "through": [2, 3, 4, 15], "onc": [2, 3, 4], "descend": [2, 3, 4, 11], "represent": [2, 3, 4], "replac": [2, 3, 4, 5], "attempt": [2, 3, 4], "complet": [2, 3, 4, 8], "contrast": [2, 3, 4], "subclass": [2, 3, 4], "transfer": [2, 3, 4], "to_undirect": [2, 3, 4], "both": [2, 3, 4, 12], "arbitrari": [2, 3, 4], "choic": [2, 3, 4], "manual": [2, 3, 4], "desir": [2, 3, 4], "g2": [2, 3, 4], "final": [2, 3, 4], "To": [2, 3, 4, 15, 16, 17], "treat": [2, 3, 4], "should": [2, 3, 4, 14], "yield": [2, 3, 4], "first": [2, 3, 4, 12, 16], "some": [2, 3, 4, 8, 17], "unless": [2, 3, 4], "introduct": [2, 3, 4, 17], "how": [2, 3, 4, 14, 15, 17], "them": [2, 3, 4, 7, 17], "partial": [3, 4, 5, 15], "uncertainti": [3, 5], "admg": [3, 4, 13, 14, 15], "implicit": [3, 4], "score": [3, 4], "ge": [3, 4], "approach": [3, 4], "suspect": [3, 4], "excluded_tripl": [3, 4], "unfaith": [3, 4], "tripl": [3, 4, 8], "frozenset": [3, 4], "mark_unfaithful_tripl": [3, 4], "v_i": [3, 4], "v_j": [3, 4], "mark": [3, 4], "third": [3, 4, 12], "simpli": [3, 4], "point": [3, 4, 5], "possible_children": [3, 4], "possible_par": [3, 4], "incoming_circle_edg": 4, "ancestr": [4, 5, 15], "term": 4, "essenti": 4, "extend": 4, "definit": [4, 9], "circular": [4, 8], "cpdag": [4, 7, 13, 20], "circle_edg": 4, "qualifi": 4, "sub_circle_graph": 4, "directed_edge_typ": 5, "bidirected_edge_typ": 5, "acyclifi": 5, "cyclic": [5, 20], "appli": 5, "call": 5, "aci": 5, "cycl": [5, 7], "whether": [5, 6, 7], "place": 5, "strongli": 5, "fulli": 5, "Then": [5, 16], "sc": 5, "made": 5, "jori": 5, "mooij": 5, "tom": 5, "claassen": 5, "us": [5, 12, 15, 16], "jona": 5, "david": 5, "sontag": 5, "editor": 5, "proceed": 5, "36th": 5, "confer": 5, "artifici": 5, "intellig": 5, "uai": 5, "124": 5, "machin": 5, "research": [5, 15], "1159": 5, "1168": 5, "pmlr": 5, "03": 5, "06": 5, "aug": 5, "2020": 5, "url": [5, 8], "mlr": 5, "v124": 5, "mooij20a": 5, "pag": [6, 7, 8, 9, 10, 11, 12, 13, 15, 20], "max_path_length": [6, 8, 9, 12], "find": [6, 8], "discrimin": 6, "least": 6, "vertex": 6, "maximum": [6, 8, 9, 12], "distanc": [6, 12], "1000": [6, 8, 9, 12], "explored_nod": 6, "found_discriminating_path": 6, "wa": 6, "disc_path": 6, "start": [6, 9, 10, 11, 12], "on_error": 7, "valid": 7, "most": 7, "compliant": 7, "node_x": [8, 9], "node_i": [8, 9], "length": [8, 9, 13, 14], "dsep": 8, "separt": 8, "along": 8, "characterist": 8, "subpath": 8, "triangl": 8, "meet": 8, "due": 8, "fact": 8, "shield": 8, "diego": 8, "colombo": 8, "marlo": 8, "maathui": 8, "marku": 8, "kalisch": 8, "thoma": 8, "richardson": 8, "high": 8, "dimension": 8, "annal": 8, "statist": 8, "40": 8, "294": 8, "321": 8, "2012": 8, "1214": 8, "11": 8, "aos940": 8, "comput": [9, 12], "end": 9, "smaller": 9, "subset": 9, "intersect": 9, "biconnect": 9, "first_nod": 12, "second_nod": 12, "uncov": 12, "potenti": 12, "opposit": 12, "addit": 12, "mean": 12, "previou": 12, "befor": 12, "henc": 12, "cannot": 12, "after": 12, "travers": 12, "its": 12, "case": 12, "rel": 12, "come": 12, "arr": [13, 14], "_supportsarrai": [13, 14], "_nestedsequ": [13, 14], "byte": [13, 14], "arr_idx": [13, 14], "graph_typ": 13, "arraylik": [13, 14], "shape": [13, 14], "n_node": [13, 14], "index": [13, 15], "format": 14, "node_ord": 14, "output": 14, "As": 14, "09": 14, "05": 14, "2022": 14, "explicitli": 14, "support": [14, 16], "moreov": 14, "simultan": 14, "although": 14, "principl": 14, "packag": 15, "known": 15, "top": 15, "maintain": 15, "test": [15, 20], "effici": [15, 18, 19, 20], "walk": 15, "encourag": 15, "your": 15, "pull": 15, "contribut": [15, 20], "guid": 15, "instal": 15, "api": [15, 18, 19, 20], "releas": 15, "histori": 15, "develop": [15, 20], "about": 15, "who": [15, 20], "codebas": 15, "contributor": 15, "page": [15, 18], "pip": 16, "avail": 16, "pypi": 16, "project": [16, 20], "poetri": 16, "recommend": 16, "repositori": 16, "git": 16, "cd": 16, "abl": 17, "everyth": 17, "concept": 17, "demonstr": 17, "auto_examples_python": 17, "auto_examples_jupyt": 17, "major": [18, 19, 20], "featur": [18, 19, 20], "enhanc": [18, 19, 20], "fix": [18, 19, 20], "link": 18, "tip": 18, "subscrib": 18, "io": 18, "notifi": 18, "adam": 20, "li": 20, "possible_ancestor": 20, "possible_descend": 20, "discriminating_path": 20, "pds_path": 20, "uncovered_pd_path": 20, "arrai": 20, "wrapper": 20, "16": 20, "acyclif": 20, "17": 20, "thank": 20, "everyon": 20, "mainten": 20, "improv": 20, "incept": 20}, "objects": {"": [[0, 0, 0, "-", "pywhy_graphs"]], "pywhy_graphs": [[2, 1, 1, "", "ADMG"], [3, 1, 1, "", "CPDAG"], [4, 1, 1, "", "PAG"]], "pywhy_graphs.ADMG": [[2, 2, 1, "", "__contains__"], [2, 2, 1, "", "__getitem__"], [2, 2, 1, "", "__iter__"], [2, 2, 1, "", "__len__"], [2, 2, 1, "", "add_edge"], [2, 2, 1, "", "add_edges_from"], [2, 3, 1, "", "adj"], [2, 3, 1, "", "bidirected_edge_name"], [2, 3, 1, "", "bidirected_edges"], [2, 2, 1, "", "c_components"], [2, 2, 1, "", "children"], [2, 2, 1, "", "clear_edge_types"], [2, 2, 1, "", "copy"], [2, 2, 1, "", "degree"], [2, 3, 1, "", "directed_edge_name"], [2, 3, 1, "", "directed_edges"], [2, 2, 1, "", "edges"], [2, 2, 1, "", "get_edge_data"], [2, 2, 1, "", "get_graphs"], [2, 4, 1, "", "graph_attr_dict_factory"], [2, 2, 1, "", "has_edge"], [2, 2, 1, "", "has_node"], [2, 2, 1, "", "is_directed"], [2, 2, 1, "", "is_multigraph"], [2, 3, 1, "", "name"], [2, 2, 1, "", "nbunch_iter"], [2, 4, 1, "", "node_attr_dict_factory"], [2, 4, 1, "", "node_dict_factory"], [2, 2, 1, "", "number_of_edge_types"], [2, 2, 1, "", "number_of_edges"], [2, 2, 1, "", "number_of_nodes"], [2, 2, 1, "", "order"], [2, 2, 1, "", "parents"], [2, 2, 1, "", "predecessors"], [2, 2, 1, "", "remove_edge"], [2, 2, 1, "", "remove_edges_from"], [2, 2, 1, "", "size"], [2, 2, 1, "", "sub_bidirected_graph"], [2, 2, 1, "", "sub_directed_graph"], [2, 2, 1, "", "sub_undirected_graph"], [2, 2, 1, "", "subgraph"], [2, 2, 1, "", "successors"], [2, 2, 1, "", "to_directed"], [2, 2, 1, "", "to_undirected"], [2, 3, 1, "", "undirected_edge_name"], [2, 3, 1, "", "undirected_edges"], [2, 2, 1, "", "update"]], "pywhy_graphs.CPDAG": [[3, 2, 1, "", "__contains__"], [3, 2, 1, "", "__getitem__"], [3, 2, 1, "", "__iter__"], [3, 2, 1, "", "__len__"], [3, 2, 1, "", "add_edge"], [3, 2, 1, "", "add_edges_from"], [3, 3, 1, "", "adj"], [3, 2, 1, "", "children"], [3, 2, 1, "", "clear_edge_types"], [3, 2, 1, "", "copy"], [3, 2, 1, "", "degree"], [3, 3, 1, "", "directed_edge_name"], [3, 3, 1, "", "directed_edges"], [3, 2, 1, "", "edges"], [3, 3, 1, "", "excluded_triples"], [3, 2, 1, "", "get_edge_data"], [3, 2, 1, "", "get_graphs"], [3, 4, 1, "", "graph_attr_dict_factory"], [3, 2, 1, "", "has_edge"], [3, 2, 1, "", "has_node"], [3, 2, 1, "", "is_directed"], [3, 2, 1, "", "is_multigraph"], [3, 2, 1, "", "mark_unfaithful_triple"], [3, 3, 1, "", "name"], [3, 2, 1, "", "nbunch_iter"], [3, 4, 1, "", "node_attr_dict_factory"], [3, 4, 1, "", "node_dict_factory"], [3, 3, 1, "", "nodes"], [3, 2, 1, "", "number_of_edge_types"], [3, 2, 1, "", "number_of_edges"], [3, 2, 1, "", "number_of_nodes"], [3, 2, 1, "", "order"], [3, 2, 1, "", "orient_uncertain_edge"], [3, 2, 1, "", "parents"], [3, 2, 1, "", "possible_children"], [3, 2, 1, "", "possible_parents"], [3, 2, 1, "", "predecessors"], [3, 2, 1, "", "remove_edge"], [3, 2, 1, "", "remove_edges_from"], [3, 2, 1, "", "size"], [3, 2, 1, "", "sub_directed_graph"], [3, 2, 1, "", "sub_undirected_graph"], [3, 2, 1, "", "subgraph"], [3, 2, 1, "", "successors"], [3, 2, 1, "", "to_directed"], [3, 2, 1, "", "to_undirected"], [3, 3, 1, "", "undirected_edge_name"], [3, 3, 1, "", "undirected_edges"], [3, 2, 1, "", "update"]], "pywhy_graphs.PAG": [[4, 2, 1, "", "__contains__"], [4, 2, 1, "", "__getitem__"], [4, 2, 1, "", "__iter__"], [4, 2, 1, "", "__len__"], [4, 2, 1, "", "add_edge"], [4, 2, 1, "", "add_edges_from"], [4, 3, 1, "", "adj"], [4, 3, 1, "", "bidirected_edge_name"], [4, 3, 1, "", "bidirected_edges"], [4, 2, 1, "", "c_components"], [4, 2, 1, "", "children"], [4, 3, 1, "", "circle_edges"], [4, 2, 1, "", "clear_edge_types"], [4, 2, 1, "", "copy"], [4, 2, 1, "", "degree"], [4, 3, 1, "", "directed_edge_name"], [4, 3, 1, "", "directed_edges"], [4, 2, 1, "", "edges"], [4, 3, 1, "", "excluded_triples"], [4, 2, 1, "", "get_edge_data"], [4, 2, 1, "", "get_graphs"], [4, 4, 1, "", "graph_attr_dict_factory"], [4, 2, 1, "", "has_edge"], [4, 2, 1, "", "has_node"], [4, 2, 1, "", "is_directed"], [4, 2, 1, "", "is_multigraph"], [4, 2, 1, "", "mark_unfaithful_triple"], [4, 3, 1, "", "name"], [4, 2, 1, "", "nbunch_iter"], [4, 4, 1, "", "node_attr_dict_factory"], [4, 4, 1, "", "node_dict_factory"], [4, 3, 1, "", "nodes"], [4, 2, 1, "", "number_of_edge_types"], [4, 2, 1, "", "number_of_edges"], [4, 2, 1, "", "number_of_nodes"], [4, 2, 1, "", "order"], [4, 2, 1, "", "orient_uncertain_edge"], [4, 2, 1, "", "parents"], [4, 2, 1, "", "possible_children"], [4, 2, 1, "", "possible_parents"], [4, 2, 1, "", "predecessors"], [4, 2, 1, "", "remove_edge"], [4, 2, 1, "", "remove_edges_from"], [4, 2, 1, "", "size"], [4, 2, 1, "", "sub_bidirected_graph"], [4, 2, 1, "", "sub_circle_graph"], [4, 2, 1, "", "sub_directed_graph"], [4, 2, 1, "", "sub_undirected_graph"], [4, 2, 1, "", "subgraph"], [4, 2, 1, "", "successors"], [4, 2, 1, "", "to_directed"], [4, 2, 1, "", "to_undirected"], [4, 3, 1, "", "undirected_edge_name"], [4, 3, 1, "", "undirected_edges"], [4, 2, 1, "", "update"]], "pywhy_graphs.algorithms": [[5, 5, 1, "", "acyclification"], [6, 5, 1, "", "discriminating_path"], [7, 5, 1, "", "is_valid_mec_graph"], [8, 5, 1, "", "pds"], [9, 5, 1, "", "pds_path"], [10, 5, 1, "", "possible_ancestors"], [11, 5, 1, "", "possible_descendants"], [12, 5, 1, "", "uncovered_pd_path"]], "pywhy_graphs.array": [[13, 5, 1, "", "clearn_arr_to_graph"], [14, 5, 1, "", "graph_to_arr"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:property", "4": "py:attribute", "5": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "property", "Python property"], "4": ["py", "attribute", "Python attribute"], "5": ["py", "function", "Python function"]}, "titleterms": {"api": 0, "most": 0, "us": [0, 1, 2, 3, 4, 17], "class": 0, "algorithm": [0, 5, 6, 7, 8, 9, 10, 11, 12], "markov": 0, "equival": 0, "convers": 0, "between": 0, "other": 0, "packag": 0, "": [0, 20], "causal": [0, 1], "graph": [0, 1, 15, 17], "an": 1, "introduct": 1, "how": 1, "them": 1, "structur": 1, "model": 1, "simul": 1, "some": 1, "data": 1, "direct": 1, "ayclic": 1, "dag": 1, "also": 1, "known": 1, "bayesian": 1, "network": 1, "acycl": 1, "mix": 1, "admg": [1, 2], "complet": 1, "partial": 1, "cpdag": [1, 3], "ancestr": 1, "pag": [1, 4], "refer": 1, "pywhy_graph": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "exampl": [2, 3, 4, 17], "acyclif": 5, "discriminating_path": 6, "is_valid_mec_graph": 7, "pd": 8, "pds_path": 9, "possible_ancestor": 10, "possible_descend": 11, "uncovered_pd_path": 12, "arrai": [13, 14], "clearn_arr_to_graph": 13, "graph_to_arr": 14, "pywhi": [15, 17], "content": 15, "get": 15, "start": 15, "team": 15, "indic": 15, "tabl": 15, "instal": 16, "releas": 18, "histori": 18, "what": 20, "new": 20, "version": 20, "0": 20, "1": 20, "changelog": 20, "code": 20, "document": 20, "contributor": 20}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinxcontrib.bibtex": 9, "nbsphinx": 4, "sphinx": 56}})