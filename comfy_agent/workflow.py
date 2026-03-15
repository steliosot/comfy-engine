import json
import requests
import uuid

from .node import Node
from .refs import DataRef


class Workflow:
    """Spark-style lazy DAG builder for ComfyUI"""

    def __init__(self, comfy_url):
        self.url = comfy_url.rstrip("/")
        r = requests.get(f"{self.url}/object_info")
        r.raise_for_status()
        self.registry = r.json()
        self.nodes = []
        self.next_id = 1
        print("Loaded nodes:", len(self.registry))

    def _resolve(self, value):
        if isinstance(value, DataRef):
            # Comfy API link format
            return [str(value.node_id), int(value.output_index)]

        if isinstance(value, list):
            return [self._resolve(v) for v in value]

        if isinstance(value, tuple):
            return [self._resolve(v) for v in value]

        if isinstance(value, dict):
            return {k: self._resolve(v) for k, v in value.items()}

        return value

    def _add_node(self, class_type, **kwargs):

        node_id = str(self.next_id)
        self.next_id += 1

        # store inputs lazily (do not resolve DataRef yet)
        node = Node(node_id, class_type, kwargs)
        self.nodes.append(node)

        # inspect outputs from Comfy registry
        outputs = self.registry.get(class_type, {}).get("output", [])

        # create DataRef objects for each output
        refs = tuple(DataRef(node_id, i) for i in range(len(outputs)))

        if len(refs) == 0:
            return None

        if len(refs) == 1:
            return refs[0]

        # multi-output nodes return tuple
        return refs

    def __getattr__(self, name):
        for node_name in self.registry:
            if node_name.lower() == name.lower():
                def wrapper(**kwargs):
                    return self._add_node(node_name, **kwargs)
                return wrapper
        raise AttributeError(name)

    def node(self, name, **kwargs):
        return self._add_node(name, **kwargs)

    def _build_dag(self):
        dag = {}
        for node in self.nodes:
            dag[node.node_id] = {
                "class_type": node.class_type,
                "inputs": {k: self._resolve(v) for k, v in node.inputs.items()},
            }
        return dag

    def validate(self):
        if not self.nodes:
            raise RuntimeError("Workflow is empty")

        produced = set()

        for node in self.nodes:
            for value in node.inputs.values():
                self._validate_refs(value, produced)
            produced.add(node.node_id)

        print("Workflow validation passed")

    def _validate_refs(self, value, produced):
        if isinstance(value, DataRef):
            if value.node_id not in produced:
                raise RuntimeError(
                    f"Invalid reference: node {value.node_id} not defined before use"
                )
            return

        if isinstance(value, (list, tuple)):
            for v in value:
                self._validate_refs(v, produced)
            return

        if isinstance(value, dict):
            for v in value.values():
                self._validate_refs(v, produced)

    def run(self, debug=False):
        self.validate()
        dag = self._build_dag()

        payload = {
            "prompt": dag,
            "client_id": str(uuid.uuid4())
        }

        if debug:
            print(json.dumps(payload, indent=2))

        r = requests.post(f"{self.url}/prompt", json=payload)
        r.raise_for_status()
        print(r.json())