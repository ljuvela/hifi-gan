import torch


class DiscriminatorMetrics():

    def __init__(self):
        
        self.true_accept_total = 0
        self.true_reject_total = 0
        self.false_accept_total = 0
        self.false_reject_total = 0
        self.num_samples_total = 0

    @property
    def accuracy(self):
        TA = self.true_accept_total
        TR = self.true_reject_total
        N = self.num_samples_total
        return 1.0 * (TA + TR) / N

    @property
    def false_accept_rate(self):
        return 1.0 * self.false_accept_total / self.num_samples_total

    @property
    def false_reject_rate(self):
        return 1.0 * self.false_reject_total / self.num_samples_total

    @property
    def equal_error_rate(self):
        return 0.5 * (self.false_rec)

    def accumulate(self, disc_real_outputs, disc_generated_outputs):
        """ 
        Args:
            disc_real_outputs: 
                shape is (batch, channels, timesteps)
            disc_generated_outputs
                shape is (batch, channels, timesteps)
        """
        pred_real = []
        pred_gen = []
        # classifications for each discriminator
        for d_real, d_gen in zip(disc_real_outputs, disc_generated_outputs):
            # mean prediction over time and channels
            pred_real.append(torch.mean(d_real, dim=(-1,)) > 0.5)
            pred_gen.append(torch.mean(d_gen, dim=(-1,)) < 0.5)

        # Stack classifications from different discriminators
        pred_real = torch.stack(pred_real, dim=0)
        pred_gen = torch.stack(pred_gen, dim=0)

        # Majority vote (probabilites not available)
        pred_real_voted, _ = torch.median(pred_real * 1.0, dim=-1)
        pred_gen_voted, _ = torch.median(pred_gen * 1.0, dim=-1)

        if pred_real_voted.shape != pred_gen_voted.shape:
            raise ValueError("Real and generated batch sizes must match")

        N = pred_real_voted.shape[0] + pred_gen_voted.shape[0]

        # True Accept
        TA = pred_real_voted.sum() 

        # True Reject
        TR = pred_gen_voted.sum()

        # False Accept
        FA = N - TR

        # False Reject
        FR = N - TA

        self.true_accept_total += TA
        self.true_reject_total += TR
        self.false_accept_total += FA
        self.false_reject_total += FR
        self.num_samples_total += N
