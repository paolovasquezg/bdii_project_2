import struct

class Record:
    def __init__(self, schema, format: str, values):
        # Normalizar schema a formato lista (similar a utils.py)
        if isinstance(schema, dict):
            # Convertir dict a lista de campos
            self.schema = [{"name": name, **spec} for name, spec in schema.items()]
        elif isinstance(schema, (list, tuple)):
            # Ya está en formato lista
            self.schema = list(schema)
        else:
            raise TypeError(f"Unsupported schema type: {type(schema)}")
        
        # Inicializar fields
        self.fields = {field["name"]: None for field in self.schema}
        self.fields.update(values)
        self.format = format

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, value):
        self.fields[key] = value

    def pack(self):
        values = []
        for field in self.schema:
            name = field["name"]
            ftype = field["type"]
            length = field.get("length", 1)
            val = self.fields[name]
            t = ftype.lower()
            if t in ("i", "int", "integer"):
                val = int(val or 0)
            elif t in ("h", "smallint"):
                val = int(val or 0)
            elif t in ("q", "bigint"):
                val = int(val or 0)
            elif t in ("f", "float", "real"):
                val = float(val or 0.0)
            elif t in ("d", "double", "double precision"):
                val = float(val or 0.0)
            elif t in ("c", "char"):
                val = (
                    val or b"" if isinstance(val, bytes) else str(val or "").encode("utf-8")
                ).ljust(length, b"\x00")
            elif t in ("s", "varchar", "string"):
                val = (
                    val or b"" if isinstance(val, bytes) else str(val or "").encode("utf-8")
                ).ljust(length, b"\x00")
            elif t in ("b", "bool", "boolean", "?"):
                val = bool(val)
            elif t in ("blob", "binary"):
                val = (val or b"").ljust(length, b"\x00")
            elif t in ("date", "datetime"):
                val = (
                    val or b"" if isinstance(val, bytes) else str(val or "").encode("utf-8")
                ).ljust(length, b"\x00")
            elif t in ("array", "list", "vector"):
                # Manejar arrays: convertir a bytes binarios
                import numpy as np
                if val is None:
                    val = [0.0] * length
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                elif not isinstance(val, list):
                    val = [float(val)] + [0.0] * (length - 1)
                
                # Asegurar que tiene exactamente la longitud correcta
                if len(val) < length:
                    val.extend([0.0] * (length - len(val)))
                elif len(val) > length:
                    val = val[:length]
                
                # Convertir a bytes usando numpy
                array_np = np.array(val, dtype=np.float32)
                val = array_np.tobytes()
                # El tamaño debe ser exactamente length * 4 bytes
                if len(val) < length * 4:
                    val += b'\x00' * (length * 4 - len(val))
                elif len(val) > length * 4:
                    val = val[:length * 4]
            values.append(val)
        return struct.pack(self.format, *values)

    @classmethod
    def unpack(cls, data, format, schema):
        # Normalizar schema a formato lista (igual que en __init__)
        if isinstance(schema, dict):
            # Convertir dict a lista de campos
            schema_list = [{"name": name, **spec} for name, spec in schema.items()]
        elif isinstance(schema, (list, tuple)):
            # Ya está en formato lista
            schema_list = list(schema)
        else:
            raise TypeError(f"Unsupported schema type: {type(schema)}")
            
        unpacked = struct.unpack(format, data)
        values = {}
        unpacked_idx = 0
        
        for field in schema_list:
            name = field["name"]
            ftype = field["type"]
            length = field.get("length", 1)
            t = ftype.lower()
            
            if t in ("array", "list", "vector"):
                # Leer datos binarios como array
                import numpy as np
                raw_bytes = unpacked[unpacked_idx]
                # Convertir bytes de vuelta a array de floats
                array_values = np.frombuffer(raw_bytes, dtype=np.float32)
                # Asegurar que tiene exactamente la longitud esperada
                if len(array_values) > length:
                    array_values = array_values[:length]
                elif len(array_values) < length:
                    # Rellenar con ceros si es necesario
                    padding = np.zeros(length - len(array_values), dtype=np.float32)
                    array_values = np.concatenate([array_values, padding])
                values[name] = array_values
                unpacked_idx += 1
            else:
                raw = unpacked[unpacked_idx]
                if t in ("i", "int", "integer"):
                    values[name] = int(raw)
                elif t in ("h", "smallint"):
                    values[name] = int(raw)
                elif t in ("q", "bigint"):
                    values[name] = int(raw)
                elif t in ("f", "float", "real"):
                    values[name] = float(raw)
                elif t in ("d", "double", "double precision"):
                    values[name] = float(raw)
                elif t in ("c", "char"):
                    values[name] = raw.decode("utf-8", errors="replace").rstrip("\x00 ")
                elif t in ("s", "varchar", "string"):
                    values[name] = raw.decode("utf-8", errors="replace").rstrip("\x00 ")
                elif t in ("b", "bool", "boolean", "?"):
                    values[name] = bool(raw)
                elif t in ("blob", "binary"):
                    values[name] = raw
                elif t in ("date", "datetime"):
                    values[name] = raw.decode("utf-8", errors="replace").rstrip("\x00 ")
                else:
                    values[name] = raw
                unpacked_idx += 1
                
        return cls(schema_list, format, values)

    def __str__(self):
        parts = []
        for f in self.schema:
            name = f["name"]
            parts.append(f"{name}: {self.fields[name]!r}")
        return "Record({ " + ", ".join(parts) + " })"

    __repr__ = __str__
