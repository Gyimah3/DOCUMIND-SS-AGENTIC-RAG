/**
 * Document management logic for DocuMind.
 */

const API_BASE = '/api/v1';
const token = localStorage.getItem('access_token');

if (!token) {
    window.location.href = '/rag';
}

const elements = {
    userEmail: document.getElementById('userEmail'),
    indexFilter: document.getElementById('indexFilter'),
    docsList: document.getElementById('docsList'),
    docsTable: document.getElementById('docsTable'),
    emptyState: document.getElementById('emptyState'),
    batchActions: document.getElementById('batchActions'),
    deleteAllBtn: document.getElementById('deleteAllBtn')
};

function decodeToken(token) {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    } catch (e) {
        return null;
    }
}

const payload = decodeToken(token);
if (payload) {
    elements.userEmail.textContent = payload.sub || payload.email || '';
}

// 1. Load Indexes
async function loadIndexes() {
    try {
        const res = await fetch(`${API_BASE}/vectorstore/list-indexes`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const indexes = await res.json();

        indexes.forEach(idx => {
            const opt = document.createElement('option');
            opt.value = idx.index_name;
            opt.textContent = idx.index_name;
            elements.indexFilter.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load indexes:', err);
    }
}

// 2. Load Documents for Index
async function loadDocuments(indexName) {
    if (!indexName) {
        elements.docsList.innerHTML = '';
        elements.batchActions.classList.add('hidden');
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/documents/${indexName}/list`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const documents = await res.json();

        renderDocuments(documents);
        elements.batchActions.classList.remove('hidden');
    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

function renderDocuments(documents) {
    elements.docsList.innerHTML = '';

    if (documents.length === 0) {
        elements.docsTable.classList.add('hidden');
        elements.emptyState.classList.remove('hidden');
        return;
    }

    elements.docsTable.classList.remove('hidden');
    elements.emptyState.classList.add('hidden');

    documents.forEach(doc => {
        const row = document.createElement('tr');
        const date = new Date(doc.created_at).toLocaleDateString();

        row.innerHTML = `
            <td><strong>${doc.title}</strong></td>
            <td><code>${doc.mimetype || 'unknown'}</code></td>
            <td>${date}</td>
            <td>
                <a href="${doc.s3_url}" target="_blank" class="btn btn-secondary btn-sm">View</a>
                <button onclick="deleteDocument('${doc.id}')" class="btn btn-delete btn-sm">Delete</button>
            </td>
        `;
        elements.docsList.appendChild(row);
    });
}

// 3. Delete Single Document
async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document? This will remove its embeddings from the vector store.')) return;

    try {
        const res = await fetch(`${API_BASE}/vectorstore/document/${docId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (res.ok) {
            loadDocuments(elements.indexFilter.value);
        } else {
            alert('Failed to delete document');
        }
    } catch (err) {
        console.error('Delete failed:', err);
    }
}

// 4. Delete All in Index
async function deleteAllInIndex() {
    const indexName = elements.indexFilter.value;
    if (!indexName) return;

    if (!confirm(`WARNING: This will delete ALL documents and their embeddings in index "${indexName}". This action cannot be undone. Continue?`)) return;

    try {
        const res = await fetch(`${API_BASE}/documents/${indexName}/delete-all`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (res.ok) {
            elements.indexFilter.value = '';
            loadDocuments('');
            // Refresh index list as the VectorStore might have been deleted too
            elements.indexFilter.innerHTML = '<option value="">Select Index...</option>';
            loadIndexes();
            alert('Index and all associated records deleted successfully.');
        } else {
            const data = await res.json();
            alert(`Failed: ${data.detail || 'deletion failed'}`);
        }
    } catch (err) {
        console.error('Bulk delete failed:', err);
    }
}

elements.indexFilter.onchange = (e) => loadDocuments(e.target.value);
elements.deleteAllBtn.onclick = deleteAllInIndex;

// Init
loadIndexes();
