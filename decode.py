import subprocess
import tempfile
import os

class BPEDecoder:
    def __init__(self, codes_file):
        self.codes_file = codes_file
    
    def decode_tokens(self, tokens):
        """Decode BPE tokens using subword-nmt"""
        if isinstance(tokens, list):
            text = ' '.join(tokens)
        else:
            text = tokens
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text + '\n')
            temp_file = f.name
        
        try:
            # Use subword-nmt to decode
            result = subprocess.run([
                'python', '-m', 'subword_nmt.apply_bpe',
                '--codes', self.codes_file,
                '--input', temp_file,
                '--output', temp_file + '.out'
            ], capture_output=True, text=True)
            
            # Read result
            with open(temp_file + '.out', 'r') as f:
                decoded = f.read().strip()
            
            # Clean up
            os.unlink(temp_file)
            os.unlink(temp_file + '.out')
            
            return decoded
        except Exception as e:
            os.unlink(temp_file)
            return self.manual_decode(text)
    
    def manual_decode(self, text):
        """Fallback manual decoding"""
        # Remove @@ markers
        text = text.replace('@@ ', '')
        # Remove special tokens
        text = ' '.join([t for t in text.split() if not t.startswith('<')])
        return text

# Usage
#decoder = BPEDecoder('wmt14_en_de/code')
#tokens = ["Und", "spar@@", "sam", "ist"]
#decoded = decoder.decode_tokens(tokens)
#print(decoded)