from models.im2im.im2im_parts import *


#ADDED TO INCLUDE IM2IM
class Im2Im(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, image_size, bilinear=True):
        super(Im2Im, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_middle = 32
        self.n_channels_out = n_channels_out
        self.image_size = image_size
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # path 1
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # joined path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, self.n_channels_middle)

    def forward(self, x):
        # Handle input with different number of channels
        if x.shape[1] != self.n_channels_in:
            # If input has fewer channels than expected, repeat the channels
            if x.shape[1] < self.n_channels_in:
                # Repeat the channels to match the expected number
                x = x.repeat(1, self.n_channels_in // x.shape[1] + 1, 1, 1)
                x = x[:, :self.n_channels_in, :, :]
            # If input has more channels than expected, take the first n_channels_in
            else:
                x = x[:, :self.n_channels_in, :, :]
                
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        return x 