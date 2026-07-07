# Top multigram units per locale

Train split matches compare_g2p (seed=42, max_test=2000). Letter/phone spans from locale ``config.json`` multigram section. EM iterations=15. Config joins **on**.

Only units where letter or phone side has length >= 2 are shown.

```

=== es_MX (43584 train pairs) ===
  multigram spans: letter=2 phone=2 (locale config)
  joins in config: letters=[('c', 'h'), ('l', 'l'), ('r', 'r')] phones=[('k', 's')]
  top 30 multigram units (n>1):
    letters        phones                 mass     letter_join? phone_join?
    t a            t a                     0.0219
    c o            k o                     0.0207
    c a            k a                     0.0184
    t e            t e                     0.0183
    d o            ð o                     0.0163
    e n            e n                     0.0151
    e s            e s                     0.0146
    r a            ɾ a                     0.0145
    t i            t i                     0.0127
    d a            ð a                     0.0126
    o n            o n                     0.0126
    d e            d e                     0.0123
    c i            s j                     0.0120
    a n            a n                     0.0118
    r e            r e                     0.0117
    t o            t o                     0.0113
    a r            a ɾ                     0.0104
    m e            m e                     0.0097
    i n            i n                     0.0095
    m a            m a                     0.0094
    l a            l a                     0.0094
    l i            l i                     0.0091
    m o            m o                     0.0088
    l e            l e                     0.0086
    t r            t ɾ                     0.0083
    o s            o s                     0.0082
    r i            ɾ i                     0.0080
    s e            s e                     0.0075
    p e            p e                     0.0073
    p r            p ɾ                     0.0072

=== fr_FR (426675 train pairs) ===
  multigram spans: letter=3 phone=2 (locale config)
  joins in config: letters=[('a', 'i', 'l'), ('a', 'u'), ('c', 'h'), ('e', 'a', 'u'), ('g', 'n'), ('i', 'l', 'l'), ('o', 'u'), ('p', 'h'), ('q', 'u')] phones=[('k', 's'), ('ɡ', 'z')]
  top 30 multigram units (n>1):
    letters        phones                 mass     letter_join? phone_join?
    e r            ə ʁ                     0.0409
    a s s          a s                     0.0215
    d é            d e                     0.0180
    e s                                    0.0152
    a i            ɛ                       0.0147
    i s            i z                     0.0143
    o n s          ɔ̃                      0.0125
    e n t                                  0.0125
    o n s          ɔ̃ z                    0.0125
    e z            e z                     0.0117
    e n t          t                       0.0115
    e s            z                       0.0094
    r e            ʁ ə                     0.0094
    a l            a l                     0.0083
    o n n          ɔ n                     0.0083
    o r            ɔ ʁ                     0.0073
    i n            i n                     0.0071
    t r            t ʁ                     0.0070
    i e z          j e                     0.0068
    r é            ʁ e                     0.0067
    e r            ɛ ʁ                     0.0066
    â t            a t                     0.0065
    e s            ə z                     0.0057
    a n t          ɑ̃ t                    0.0056
    a r            a ʁ                     0.0055
    è r            ɛ ʁ                     0.0055
    e n t          ɑ̃ t                    0.0053
    a i s          ɛ z                     0.0053
    a s            a z                     0.0053
    c o n          k ɔ̃                    0.0053

=== de_DE (312077 train pairs) ===
  multigram spans: letter=3 phone=2 (locale config)
  joins in config: letters=[('a', 'i'), ('a', 'u'), ('c', 'h'), ('c', 'k'), ('e', 'i'), ('e', 'u'), ('i', 'e'), ('p', 'f'), ('p', 'h'), ('q', 'u'), ('s', 'c', 'h'), ('t', 's', 'c', 'h'), ('ä', 'u')] phones=[('t', 's')]
  top 30 multigram units (n>1):
    letters        phones                 mass     letter_join? phone_join?
    e n            ə n                     0.0447
    e r            ə ʁ                     0.0420
    s t            s t                     0.0291
    t e            t ə                     0.0214
    e s            ə s                     0.0185
    g e            ɡ ə                     0.0142
    s t            ʃ t                     0.0110
    i₊e r          i ʁ                     0.0109
    b e            b ə                     0.0105
    e m            ə m                     0.0103
    i g            ɪ ɡ                     0.0092
    a n            a n                     0.0090
    i c₊h          ɪ x                     0.0086
    v e            f ɛ                     0.0083
    u n            ʊ n                     0.0079
    i s₊c₊h        ɪ ʃ                     0.0067
    i g            ɪ x                     0.0064
    e r            ɛ ʁ                     0.0059
    n d            n t                     0.0059
    e₊i n          aɪ n                    0.0058
    z u            t₊s u                   0.0056
    a₊u s          aʊ s                    0.0055
    u n g          ʊ ŋ                     0.0053
    e l            ə l                     0.0051
    a b            a p                     0.0051
    e n            ɛ n                     0.0050
    s p            ʃ p                     0.0048
    h a            h a                     0.0046
    a₊u f          aʊ f                    0.0046
    n d            n d                     0.0044

=== en_US (133111 train pairs) ===
  multigram spans: letter=2 phone=2 (locale config)
  joins in config: letters=[('c', 'h'), ('c', 'k'), ('n', 'g'), ('p', 'h'), ('s', 'h'), ('t', 'h'), ('w', 'h')] phones=[('j', 'oʊ'), ('j', 'u'), ('j', 'ə'), ('j', 'ʊ'), ('k', 's'), ('k', 'ʃ'), ('m', 'ə', 'k'), ('t', 's'), ('ə', 'l'), ('ə', 'm'), ('ə', 'n'), ('ɡ', 'z'), ('ɪ', 'z')]
  top 30 multigram units (n>1):
    letters        phones                 mass     letter_join? phone_join?
    e r            ɝ                       0.0338
    o n            ə₊n                     0.0117
    i n₊g          ɪ ŋ                     0.0116
    s t            s t                     0.0110
    a n            ə₊n                     0.0098
    a r            ɑ ɹ                     0.0096
    ' s            z                       0.0088
    l l            l                       0.0085
    o r            ɔ ɹ                     0.0082
    i n            ɪ n                     0.0081
    e n            ə₊n                     0.0078
    a l            ə₊l                     0.0059
    a n            æ n                     0.0057
    s s            s                       0.0056
    e n            ɛ n                     0.0045
    o r            ɝ                       0.0045
    t s            t₊s                     0.0043
    u r            ɝ                       0.0042
    t t            t                       0.0040
    l e            ə₊l                     0.0039
    i e            i                       0.0038
    r e            ɹ i                     0.0038
    t i            ʃ                       0.0036
    e d            d                       0.0036
    r o            ɹ oʊ                    0.0034
    r e            ɹ ɛ                     0.0032
    a r            ɝ                       0.0032
    n e            n                       0.0031
    e e            i                       0.0031
    e a            i                       0.0030

=== pt_BR (64143 train pairs) ===
  multigram spans: letter=2 phone=2 (locale config)
  joins in config: letters=[('c', 'h'), ('g', 'u'), ('l', 'h'), ('n', 'h'), ('q', 'u'), ('r', 'r'), ('s', 's')] phones=[]
  top 30 multigram units (n>1):
    letters        phones                 mass     letter_join? phone_join?
    e n            ẽ                       0.0202
    d o            d u                     0.0164
    t a            t a                     0.0157
    a n            ɐ̃                      0.0153
    d a            d a                     0.0151
    r i            ɾ i                     0.0140
    c a            k a                     0.0128
    r a            ɾ a                     0.0127
    d e            d e                     0.0124
    t i            tʃ i                    0.0116
    i n            ĩ                       0.0112
    o n            õ                       0.0105
    l i            l i                     0.0104
    c i            s i                     0.0095
    d i            dʒ i                    0.0087
    l a            l a                     0.0079
    r e            ʁ e                     0.0079
    a m            ɐ̃ w̃                   0.0077
    t e            tʃ i                    0.0076
    t e            t e                     0.0073
    m a            m a                     0.0071
    c o            k o                     0.0067
    o s            u s                     0.0065
    r e            ɾ e                     0.0062
    t r            t ɾ                     0.0062
    ã o            ɐ̃ w̃                   0.0061
    v a            v a                     0.0059
    p e            p e                     0.0058
    p a            p a                     0.0054
    t o            t u                     0.0054

=== it_IT (19560 train pairs) ===
  multigram spans: letter=2 phone=2 (locale config)
  joins in config: letters=[('c', 'h'), ('g', 'l', 'i'), ('g', 'n'), ('l', 'l'), ('s', 'c'), ('s', 'c', 'e'), ('s', 'c', 'i'), ('z', 'z')] phones=[('k', 's'), ('t', 's'), ('ts', 'ts'), ('ɲ', 'ɲ'), ('ʃ', 'ʃ'), ('ʎ', 'ʎ')]
  top 30 multigram units (n>1):
    letters        phones                 mass     letter_join? phone_join?
    r e            r e                     0.0324
    t o            t o                     0.0209
    c o            k o                     0.0200
    r i            r i                     0.0172
    t e            t e                     0.0170
    r a            r a                     0.0167
    t i            t i                     0.0153
    t a            t a                     0.0151
    c a            k a                     0.0147
    n e            n e                     0.0143
    s t            s t                     0.0118
    m e            m e                     0.0114
    d i            d i                     0.0112
    m a            m a                     0.0111
    r o            r o                     0.0105
    l e            l e                     0.0104
    n o            n o                     0.0100
    l i            l i                     0.0096
    i o            j o                     0.0092
    i n            i n                     0.0089
    l₊l            l l                     0.0085
    n a            n a                     0.0082
    l a            l a                     0.0081
    m o            m o                     0.0079
    d e            d e                     0.0078
    v a            v a                     0.0077
    s s            s s                     0.0075
    m i            m i                     0.0072
    d o            d o                     0.0072
    n i            n i                     0.0069
```
