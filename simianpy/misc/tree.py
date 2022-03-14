import h5py
from pathlib import Path

class Tree:
    mid, end, bare = '\u251C','\u2514','\u2502' # "├", "└", "│"
    def __repr__(self):
        return f'Tree <{self.name}> with {len(self.nodes)} children' if self.nodes else f'Node <{self.name}>'
    def __getitem__(self, key):
        key = key.strip('/')
        key, rest = key.split('/', 1) if '/' in key else (key, '')
        for node in self.nodes:
            if node.name == key:
                return node[rest] if rest else node
        else:
            raise KeyError(key, 'not found')
    def __init__(self, name, nodes=None):
        self.name = name
        self.nodes = nodes if nodes is not None else []
    def add_node(self, node):
        self.nodes.append(node)
    @classmethod
    def from_dict(cls, name, d, recurse=True, include_values=False):
        self = cls(name)
        for key, val in d.items():
            if isinstance(val, dict):
                self.add_node(Tree.from_dict(key, val, recurse=recurse))
            else:
                self.add_node(Tree(key, [val] if include_values else None))
        return self
    @classmethod
    def from_hdf5(cls, name, h5group):
        self = cls(name)
        for key, val in h5group.items():
            if isinstance(val, h5py.Group):
                self.add_node(Tree.from_hdf5(key, val))
            else:
                self.add_node(Tree(key))
        return self
    def print(self, prepend='', lastchild=True, level=0, maxdepth=None, maxchildren=None, sortnodes=False):
        print(prepend,self.end if lastchild else self.mid,self.name)
        lastchildidx = len(self.nodes)-1
        # print(prepend_, self.end if lastchild_ else self.mid, node.name, '[...]' if node.nodes else '')
        if maxdepth is None or level<maxdepth:
            nodes = sorted(self.nodes, key=lambda x: (-len(x.nodes), x.name)) if sortnodes else self.nodes
            for idx, node in enumerate(nodes):
                if maxchildren is not None and idx>=maxchildren:
                    print(prepend+' ' if lastchild else prepend+' '+self.bare,self.end,f'[{len(self.nodes)-idx} more]')
                    break
                node.print(
                    prepend+' ' if lastchild else prepend+' '+self.bare, 
                    lastchild=idx==lastchildidx, 
                    level=level+1, 
                    maxdepth=maxdepth,
                    maxchildren=maxchildren,
                    sortnodes=sortnodes
                )
        elif self.nodes:
            print(prepend+'  ' if lastchild else prepend+' '+self.bare, self.end, '[...]')