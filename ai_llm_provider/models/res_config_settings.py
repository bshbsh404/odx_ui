from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    ai_provider_count = fields.Integer(compute='_compute_ai_provider_count')

    def _compute_ai_provider_count(self):
        count = self.env['ai.llm.provider'].search_count([])
        for record in self:
            record.ai_provider_count = count
