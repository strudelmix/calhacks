# "Build a Wallet" tutorial, step 1: slightly more than "Hello World"
# This step demonstrates a simple GUI and XRPL connectivity.
# License: MIT. https://github.com/XRPLF/xrpl-dev-portal/blob/master/LICENSE
import xrpl
import streamlit as st

JSON_RPC_URL = "https://s.altnet.rippletest.net:51234/"


# get XRP Ledger account
def get_account(seed):
    # request new client from XRP Ledger
    client = xrpl.clients.JsonRpcClient(JSON_RPC_URL)

    # creates new wallet
    if (seed == ''):
        new_wallet = xrpl.wallet.generate_faucet_wallet(client)
    else:
        new_wallet = xrpl.wallet.Wallet.from_seed(seed)
    return new_wallet


def send_xrp():
    client = xrpl.clients.JsonRpcClient(JSON_RPC_URL)
    # transaction request, this would have no sending or destination amount
    # as it is a live video feed
    # check if response fails, return response
    try:
        response = client.request(xrpl.models.requests.Ledger(
            ledger_index="validated"
        ))
        if response.is_successful():
            return response.result['ledger_hash']

    except Exception as e:
        st.write(f"Failed to get validated ledger from server. ({e})")
        # st.write(f"Server returned an error: {response.result['error_message']}")

        # Connected to the server, but the request failed. This can
        # happen if, for example, the server isn't synced to the network
        # so it doesn't have the latest validated ledger.
        return False




