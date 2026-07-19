import requests
import json
import base64

def get_xrpl_example():
    print("--- 1. Extrayendo ejemplo de XRP Ledger (vía RPC Público) ---")
    url = "https://xrplcluster.com/"  L
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        ledger_index_payload = {
            "method": "ledger",
            "params": [{"ledger_index": "validated", "transactions": False}]
        }
        res = requests.post(url, json=ledger_index_payload, headers=headers)
        if res.status_code != 200:
            print("Error al conectar con el nodo XRPL.\n")
            return
            
        current_ledger = res.json()['result']['ledger_index']
        encontrado = False
        intentos = 0
        
        while not encontrado and intentos < 10:
            tx_payload = {
                "method": "ledger",
                "params": [{
                    "ledger_index": current_ledger,
                    "transactions": True,
                    "expand": True
                }]
            }
            res_tx = requests.post(url, json=tx_payload, headers=headers)
            transactions = res_tx.json().get('result', {}).get('ledger', {}).get('transactions', [])
            
            for tx in transactions:
                if isinstance(tx, dict) and 'Memos' in tx:
                    print(f"Ledger Index: {current_ledger}")
                    print(f"TX Hash: {tx.get('hash')}")
                    print(f"Tipo de Transacción: {tx.get('TransactionType')}")
                    
        
                    for memo_item in tx['Memos']:
                        memo = memo_item.get('Memo', {})
                        memo_type_hex = memo.get('MemoType', '')
                        memo_data_hex = memo.get('MemoData', '')
                        
                        print(f"MemoType (Hex Raw): {memo_type_hex}")
                        print(f"MemoData (Hex Raw): {memo_data_hex}")
                        
                        
                        try:
                            if memo_type_hex:
                                m_type = bytes.fromhex(memo_type_hex).decode('utf-8', errors='replace')
                                print(f"-> MemoType Decodificado: {m_type}")
                            if memo_data_hex:
                                m_data = bytes.fromhex(memo_data_hex).decode('utf-8', errors='replace')
                                print(f"-> MemoData Decodificado: {m_data}")
                        except Exception as e:
                            print(f"[!] Error al decodificar Hex: {e}")
                            
                    encontrado = True
                    print()
                    break
            
            current_ledger -= 1
            intentos += 1
            
        if not encontrado:
            print("No se encontraron transacciones con Memos en los últimos 10 ledgers.\n")
            
    except Exception as e:
        print(f"Error de conexión con XRPL: {e}\n")


def get_stellar_example():
    print("--- 2. Extrayendo ejemplo de Stellar (vía Horizon API) ---")
    url = "https://horizon.stellar.org/transactions?limit=50&order=desc"
    try:
        intentos = 0
        encontrado = False
        while url and not encontrado and intentos < 5:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                txs = data.get('_embedded', {}).get('records', [])
                for tx in txs:
                    if 'memo' in tx and tx.get('memo_type') != 'none':
                        print(f"TX Hash: {tx['hash']}")
                        print(f"Tipo de Memo: {tx['memo_type']}")
                        print(f"Contenido del Memo: {tx.get('memo')}\n")
                        encontrado = True
                        break
                url = data.get('_links', {}).get('next', {}).get('href')
                intentos += 1
            else:
                print(f"Error en Horizon API: {response.status_code}\n")
                break
        if not encontrado:
            print("No se encontraron transacciones con memo tras revisar registros recientes.\n")
    except Exception as e:
        print(f"Error de conexión con Stellar: {e}\n")


def get_hedera_example():
    print("--- 3. Extrayendo ejemplo de Hedera (vía Mirror Node REST API) ---")
    url = "https://mainnet-public.mirrornode.hedera.com/api/v1/transactions?limit=20&order=desc"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            txs = response.json().get('transactions', [])
            for tx in txs:
                memo_b64 = tx.get('memo_base64')
                if memo_b64:
                    print(f"ID de Transacción: {tx['transaction_id']}")
                    print(f"Memo (Base64 Raw): {memo_b64}")
                    try:
                        decoded_bytes = base64.b64decode(memo_b64)
                        decoded_text = decoded_bytes.decode('utf-8', errors='replace')
                        print(f"Memo (Decodificado Forzado): {decoded_text}\n")
                    except Exception:
                        print("Memo (Decodificado Forzado): [No se pudo decodificar]\n")
                    break
        else:
            print(f"Error en Hedera Mirror Node API: {response.status_code}\n")
    except Exception as e:
        print(f"Error de conexión con Hedera: {e}\n")


if __name__ == "__main__":
    get_xrpl_example()
    get_stellar_example()
    get_hedera_example()
