from pathlib import Path
from io import BytesIO
import struct
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import util


BASE_DIR = Path(__file__).resolve().parent

MI_MATRIX = 14
MI_COMPRESSED = 15


DATASET_ALIASES = {
    "BDGP": "BDGP",
    "Handwritten": "HandWritten",
    "HandWritten": "HandWritten",
    "Reuters": "Reuters_dim10",
    "Reuters_dim10": "Reuters_dim10",
    "Wikipedia": "Wikipedia",
}


def normalize_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def _read_data_element(stream, endian):
    tag = stream.read(8)
    if len(tag) == 0:
        return None
    if len(tag) < 8:
        raise ValueError("Invalid MAT v5 data element tag")

    first, second = struct.unpack(endian + "II", tag)
    if first >> 16:
        data_type = first & 0xFFFF
        num_bytes = first >> 16
        return data_type, tag[4:4 + num_bytes]

    data_type = first
    num_bytes = second
    data = stream.read(num_bytes)
    padding = (8 - num_bytes % 8) % 8
    if padding:
        stream.read(padding)
    return data_type, data


def _mat_dtype(data_type, endian):
    dtypes = {
        1: "i1",
        2: "u1",
        3: endian + "i2",
        4: endian + "u2",
        5: endian + "i4",
        6: endian + "u4",
        7: endian + "f4",
        9: endian + "f8",
        12: endian + "i8",
        13: endian + "u8",
    }
    if data_type not in dtypes:
        raise ValueError(f"Unsupported MAT v5 data type: {data_type}")
    return np.dtype(dtypes[data_type])


def _parse_mat_matrix(payload, endian):
    stream = BytesIO(payload)

    _, flags_data = _read_data_element(stream, endian)
    array_flags = struct.unpack(endian + "II", flags_data[:8])
    class_type = array_flags[0] & 0xFF

    dims_type, dims_data = _read_data_element(stream, endian)
    dims = np.frombuffer(dims_data, dtype=_mat_dtype(dims_type, endian)).astype(int)
    dims = tuple(int(dim) for dim in dims)

    name_type, name_data = _read_data_element(stream, endian)
    name = np.frombuffer(name_data, dtype=_mat_dtype(name_type, endian)).tobytes().decode("latin1")

    if class_type == 1:
        cells = []
        while stream.tell() < len(payload):
            element = _read_data_element(stream, endian)
            if element is None:
                break
            data_type, data = element
            if data_type == MI_MATRIX:
                _, value = _parse_mat_matrix(data, endian)
                cells.append(value)
        return name, cells

    data_type, real_data = _read_data_element(stream, endian)
    dtype = _mat_dtype(data_type, endian)
    count = int(np.prod(dims))
    array = np.frombuffer(real_data, dtype=dtype, count=count).copy()
    array = array.reshape(dims, order="F")
    return name, array


def _load_mat_v5(mat_path):
    with open(mat_path, "rb") as f:
        header = f.read(128)
        endian_indicator = header[126:128]
        endian = "<" if endian_indicator == b"IM" else ">"
        variables = {}

        while True:
            element = _read_data_element(f, endian)
            if element is None:
                break
            data_type, data = element
            if data_type == MI_COMPRESSED:
                inner = BytesIO(zlib.decompress(data))
                while True:
                    inner_element = _read_data_element(inner, endian)
                    if inner_element is None:
                        break
                    inner_type, inner_data = inner_element
                    if inner_type == MI_MATRIX:
                        name, value = _parse_mat_matrix(inner_data, endian)
                        variables[name] = value
            elif data_type == MI_MATRIX:
                name, value = _parse_mat_matrix(data, endian)
                variables[name] = value

    return variables


def _load_mat(mat_path):
    header = mat_path.read_bytes()[:128]
    if header.startswith(b"MATLAB 5.0 MAT-file"):
        return _load_mat_v5(mat_path)

    import scipy.io as sio

    return sio.loadmat(mat_path)


def _resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_absolute():
        return data_dir
    return BASE_DIR / data_dir


def _read_cell_views(x_all):
    if isinstance(x_all, list):
        return [np.asarray(view) for view in x_all]
    x_all = np.squeeze(x_all)
    return [np.asarray(x_all[view]) for view in range(x_all.shape[0])]


def _read_numbered_views(mat):
    view_keys = sorted(
        [key for key in mat.keys() if key.startswith("X") and key[1:].isdigit()],
        key=lambda item: int(item[1:]),
    )
    return [np.asarray(mat[key]) for key in view_keys]


def _read_labels(mat):
    for key in ["Y", "truth", "label", "labels", "gt"]:
        if key in mat:
            return np.squeeze(mat[key]).astype("int")
    return None


def load_data(data_name, data_dir="data"):
    data_name = normalize_dataset_name(data_name)
    if data_name == "Wikipedia":
        raise NotImplementedError(
            "Wikipedia dataset alias is registered but dataset loading is not implemented yet."
        )
    mat_path = _resolve_data_dir(data_dir) / f"{data_name}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    mat = _load_mat(mat_path)
    x_list = []

    if data_name in ["HandWritten", "Caltech101-7", "Caltech101-20", "Scene-15"]:
        x_list = _read_cell_views(mat["X"])
        y = np.squeeze(mat["Y"]).astype("int")
    elif data_name in ["aloideep3v", "NH_face"]:
        x_list = _read_cell_views(mat["X"])
        y = np.squeeze(mat["truth"]).astype("int")
    elif data_name == "Reuters_dim10":
        x_train = mat["x_train"]
        x_test = mat["x_test"]
        y = np.squeeze(np.hstack((mat["y_train"], mat["y_test"]))).astype("int")
        for view in range(x_train.shape[0]):
            x_list.append(np.vstack((x_train[view], x_test[view])))
    elif data_name == "Fashion":
        x_list.append(mat["X1"].reshape(10000, -1))
        x_list.append(mat["X2"].reshape(10000, -1))
        x_list.append(mat["X3"].reshape(10000, -1))
        y = np.squeeze(mat["Y"]).astype("int")
    elif data_name == "BDGP":
        x_list = _read_cell_views(mat["X"]) if "X" in mat else _read_numbered_views(mat)
        if len(x_list) == 0:
            raise ValueError("BDGP data must contain X cell array or X1, X2, ... arrays")
        y = _read_labels(mat)
        if y is None:
            y = np.zeros(x_list[0].shape[0], dtype="int")
    else:
        raise Exception("Undefined data_name")

    y = y - np.min(y)
    return x_list, [y]


def generate_incomplete_idx(num_sample, num_views, missing_rate, seed):
    rng = np.random.default_rng(seed)
    inc_idx = (rng.random((num_sample, num_views)) > missing_rate).astype("int32")
    empty_rows = np.where(inc_idx.sum(axis=1) == 0)[0]
    for row in empty_rows:
        inc_idx[row, rng.integers(0, num_views)] = 1
    return inc_idx


def get_incomplete_idx(data_name, missing_rate, num_sample, num_views, seed, data_dir="data"):
    data_name = normalize_dataset_name(data_name)
    mask_path = _resolve_data_dir(data_dir) / f"{data_name}_percentDel_{missing_rate}.mat"
    if not mask_path.exists():
        return generate_incomplete_idx(num_sample, num_views, missing_rate, seed)

    mat = _load_mat(mask_path)
    if "folds" in mat:
        folds_data = mat["folds"]
        inc_idx = folds_data[0, 0] if folds_data.dtype == object else folds_data
    elif "per" in mat:
        per = mat["per"]
        inc_idx = per[0, 0] if per.dtype == object else per
    elif "inc_idx" in mat:
        inc_idx = mat["inc_idx"]
    elif "mask" in mat:
        inc_idx = mat["mask"]
    else:
        raise KeyError(f"No folds/per/inc_idx/mask key found in {mask_path}")

    inc_idx = np.array(inc_idx, "int32")
    expected_shape = (num_sample, num_views)
    if inc_idx.shape != expected_shape:
        raise ValueError(f"Mask shape {inc_idx.shape} does not match data shape {expected_shape}")
    return inc_idx


def norm_data(data_name, x_list):
    data_name = normalize_dataset_name(data_name)
    if data_name == "BDGP":
        return [util.normalize(x).astype("float32") for x in x_list]

    x_list_new = []
    for view_data in x_list:
        view_data = view_data.astype(np.float32)
        mean = np.mean(view_data, axis=0, keepdims=True)
        std = np.std(view_data, axis=0, keepdims=True)
        std[std == 0] = 1
        x_list_new.append((view_data - mean) / std)
    return x_list_new


class ComDataset(Dataset):
    def __init__(self, fea, inc, device):
        self.device = device
        self.fea = fea
        self.inc = inc

    def __getitem__(self, index):
        x_list = [torch.from_numpy(x[index]).to(self.device) for x in self.fea]
        inc_mask = torch.from_numpy(self.inc[index]).to(self.device)
        return x_list, inc_mask, index

    def __len__(self):
        return self.fea[0].shape[0]


def get_loader(config, device):
    data_cfg = config["Dataset"]
    x, y = load_data(data_cfg["name"], data_cfg.get("data_dir", "data"))
    y = y[0]

    inc_idx = get_incomplete_idx(
        data_cfg["name"],
        data_cfg["missing_rate"],
        x[0].shape[0],
        len(x),
        config["training"]["seed"],
        data_cfg.get("data_dir", "data"),
    )

    x = norm_data(data_cfg["name"], x)
    masked_x = []
    for view_idx, view in enumerate(x):
        view_mask = inc_idx[:, view_idx][:, np.newaxis]
        masked_x.append((view * view_mask).astype("float32"))

    dataset = ComDataset(masked_x, inc_idx, device)
    data_loader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True)
    return data_loader, x, y, inc_idx, masked_x


class TransformerDataset2(Dataset):
    def __init__(self, emb, inc_idx, device):
        self.device = device
        self.view_num = len(emb)
        self.inc = inc_idx
        self.features = [torch.from_numpy(fea).to(self.device) for fea in emb]
        self.features = torch.stack(
            [features.unsqueeze(1) for features in self.features],
            dim=1,
        ).squeeze(2)

    def __getitem__(self, index):
        out_mask = self.inc[index]
        in_mask = torch.zeros_like(out_mask)
        while torch.sum(in_mask) == 0:
            random_mask = torch.from_numpy(np.random.randint(2, size=self.view_num)).to(self.device)
            in_mask = out_mask * random_mask

        fea = self.features[index]
        mask_fea = fea * in_mask.unsqueeze(1).expand_as(fea).float()
        return mask_fea, fea, in_mask, out_mask, index

    def __len__(self):
        return self.features.shape[0]


def get_transformer_loader2(emb, inc_idx, batch_size, device):
    dataset = TransformerDataset2(emb, inc_idx, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
