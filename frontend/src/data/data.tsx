

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

export const loadTables = async () => {
  try {
    const response = await fetch(`${API_BASE}/tables`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      return { success: false, error: response.status };
    }

    const data = await response.json();
    return { success: true, data: data };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
};

export const execQuery = async (query: {content: string}) => {
  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(query)
    });

    if (!response.ok) {
      return { success: false, error: response.status };
    }

    const data = await response.json();
    return { success: true, data: data };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
};

export const uploadMedia = async (file: File) => {
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch(`${API_BASE}/upload-media`, {
      method: 'POST',
      body: form
    });
    if (!response.ok) {
      return { success: false, error: response.status };
    }
    const data = await response.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
};
