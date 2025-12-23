from torch.nn.utils.rnn import pack_padded_sequence
from config import *

def inference_lstm(model, src, device, max_len):
    """
    src: (T, 1, 4)
    return: (max_len, 1, 4)
    """
    model.eval()

    src = src.to(device)

    with torch.no_grad():
        src_len = torch.tensor([src.shape[0]], dtype=torch.int64)

        src_packed = pack_padded_sequence(
            src,
            src_len,
            batch_first=False,
            enforce_sorted=False
        )

        _, (hidden, cell) = model.encoder(src_packed)

        decoder_input = torch.zeros(1, 1, 4, device=device)
        outputs = []

        for _ in range(max_len):
            dec_out, (hidden, cell) = model.decoder(
                decoder_input, (hidden, cell)
            )

            pred = model.output(dec_out)
            outputs.append(pred)
            decoder_input = pred

        return torch.cat(outputs, dim=0)
